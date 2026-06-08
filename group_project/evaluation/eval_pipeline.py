"""Local RAG evaluation pipeline for the group project.

The README allows DeepEval/RAGAS/TruLens, but those frameworks usually need
extra model credentials. This script implements the same four required metric
families with deterministic local scoring so the team can always produce a
repeatable report for the demo.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task10_generation import _generate_local_answer, reorder_for_llm
from src.task9_retrieval_pipeline import retrieve

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def overlap_score(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def coverage_score(needle: str, haystack: str) -> float:
    needle_tokens = tokenize(needle)
    haystack_tokens = tokenize(haystack)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def run_config(question: str, *, use_reranking: bool, threshold: float) -> dict:
    sources = retrieve(
        question,
        top_k=5,
        score_threshold=threshold,
        use_reranking=use_reranking,
    )
    ordered = reorder_for_llm(sources)
    answer = _generate_local_answer(question, ordered)
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": sources[0].get("source", "none") if sources else "none",
    }


def score_case(item: dict, result: dict) -> dict:
    answer = result["answer"]
    contexts = [source.get("content", "") for source in result.get("sources", [])]
    joined_context = "\n".join(contexts)
    expected_answer = item["expected_answer"]
    expected_context = item["expected_context"]
    question = item["question"]

    answer_relevance = overlap_score(expected_answer, answer)
    context_recall = coverage_score(expected_context, joined_context)

    if contexts:
        context_precision = statistics.mean(
            coverage_score(question, context) for context in contexts
        )
    else:
        context_precision = 0.0

    citation_bonus = 1.0 if "[" in answer and "]" in answer else 0.4
    faithfulness = min(
        1.0,
        0.7 * coverage_score(answer, joined_context) + 0.3 * citation_bonus,
    )

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_recall": round(context_recall, 3),
        "context_precision": round(context_precision, 3),
        "average": round(
            statistics.mean(
                [faithfulness, answer_relevance, context_recall, context_precision]
            ),
            3,
        ),
    }


def evaluate_config(name: str, dataset: list[dict], *, use_reranking: bool, threshold: float) -> dict:
    rows = []
    for item in dataset:
        result = run_config(
            item["question"],
            use_reranking=use_reranking,
            threshold=threshold,
        )
        scores = score_case(item, result)
        rows.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "retrieval_source": result["retrieval_source"],
                "scores": scores,
                "source_count": len(result.get("sources", [])),
            }
        )

    summary = {}
    metric_names = [
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
        "average",
    ]
    for metric in metric_names:
        summary[metric] = round(
            statistics.mean(row["scores"][metric] for row in rows),
            3,
        )

    return {"name": name, "summary": summary, "rows": rows}


def compare_configs(dataset: list[dict]) -> dict:
    # Disable external reranker during offline evaluation so results are stable.
    os.environ["JINA_API_KEY"] = ""
    return {
        "hybrid_rerank": evaluate_config(
            "Hybrid + local rerank",
            dataset,
            use_reranking=True,
            threshold=-1.0,
        ),
        "hybrid_no_rerank": evaluate_config(
            "Hybrid without rerank",
            dataset,
            use_reranking=False,
            threshold=-1.0,
        ),
    }


def metric_label(metric: str) -> str:
    return {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "Average",
    }[metric]


def worst_rows(config: dict, limit: int = 3) -> list[dict]:
    return sorted(config["rows"], key=lambda row: row["scores"]["average"])[:limit]


def export_results(comparison: dict) -> None:
    config_a = comparison["hybrid_rerank"]
    config_b = comparison["hybrid_no_rerank"]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "Local deterministic evaluation: metric heuristics tương ứng với Faithfulness, Answer Relevance, Context Recall và Context Precision. Cách này chạy offline, không phụ thuộc OpenAI/DeepEval/RAGAS runtime.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid no-rerank) | Delta |",
        "|--------|-----------------------------|------------------------------|-------|",
    ]

    for metric in [
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
        "average",
    ]:
        a_score = config_a["summary"][metric]
        b_score = config_b["summary"][metric]
        lines.append(
            f"| {metric_label(metric)} | {a_score:.3f} | {b_score:.3f} | {a_score - b_score:+.3f} |"
        )

    winner = config_a if config_a["summary"]["average"] >= config_b["summary"]["average"] else config_b
    lines.extend(
        [
            "",
            "## A/B Comparison Analysis",
            "",
            f"**Config A:** {config_a['name']} dùng semantic search + lexical BM25, merge RRF, sau đó rerank.",
            "",
            f"**Config B:** {config_b['name']} dùng semantic search + lexical BM25, merge RRF nhưng bỏ bước rerank.",
            "",
            f"**Kết luận:** {winner['name']} có điểm trung bình tốt hơn trong lần chạy này. Điểm context recall/precision nên được ưu tiên cải thiện bằng cách bổ sung dữ liệu pháp luật đã OCR đầy đủ và tune threshold fallback PageIndex.",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | Average | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |",
            "|---|----------|---------|--------------|-----------|--------|---------------|------------|",
        ]
    )

    for idx, row in enumerate(worst_rows(config_a), 1):
        scores = row["scores"]
        question = row["question"].replace("|", "/")
        lines.append(
            f"| {idx} | {question} | {scores['average']:.3f} | {scores['faithfulness']:.3f} | "
            f"{scores['answer_relevance']:.3f} | {scores['context_recall']:.3f} | Retrieval/Generation | "
            f"Expected context chưa khớp mạnh với chunks hoặc dữ liệu OCR còn thiếu. |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** OCR lại đầy đủ các PDF pháp luật để thay các placeholder markdown.",
            "**Expected impact:** Tăng context recall và faithfulness cho câu hỏi pháp luật chi tiết.",
            "",
            "### Cải tiến 2",
            "**Action:** Tune score threshold và ưu tiên PageIndex fallback cho câu hỏi pháp luật dài.",
            "**Expected impact:** Truy xuất đúng đoạn trong PDF hơn khi hybrid local index yếu.",
            "",
            "### Cải tiến 3",
            "**Action:** Khi demo có mạng ổn định, bật Gemini/Jina thay cho fallback local.",
            "**Expected impact:** Câu trả lời tự nhiên hơn và reranking chính xác hơn.",
            "",
            "## Run Metadata",
            "",
            f"- Golden dataset size: {len(config_a['rows'])}",
            "- Configs compared: 2",
            "- Metrics: 4 required metrics + average",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dataset = load_golden_dataset()
    comparison = compare_configs(dataset)
    export_results(comparison)

    print(f"Loaded {len(dataset)} test cases")
    for config in comparison.values():
        print(config["name"], config["summary"])
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
