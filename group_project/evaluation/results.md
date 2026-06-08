# RAG Evaluation Results

## Framework sử dụng

Local deterministic evaluation: metric heuristics tương ứng với Faithfulness, Answer Relevance, Context Recall và Context Precision. Cách này chạy offline, không phụ thuộc OpenAI/DeepEval/RAGAS runtime.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no-rerank) | Delta |
|--------|-----------------------------|------------------------------|-------|
| Faithfulness | 0.932 | 0.933 | -0.001 |
| Answer Relevance | 0.057 | 0.059 | -0.002 |
| Context Recall | 0.233 | 0.317 | -0.084 |
| Context Precision | 0.276 | 0.272 | +0.004 |
| Average | 0.375 | 0.395 | -0.020 |

## A/B Comparison Analysis

**Config A:** Hybrid + local rerank dùng semantic search + lexical BM25, merge RRF, sau đó rerank.

**Config B:** Hybrid without rerank dùng semantic search + lexical BM25, merge RRF nhưng bỏ bước rerank.

**Kết luận:** Hybrid without rerank có điểm trung bình tốt hơn trong lần chạy này. Điểm context recall/precision nên được ưu tiên cải thiện bằng cách bổ sung dữ liệu pháp luật đã OCR đầy đủ và tune threshold fallback PageIndex.

## Worst Performers (Bottom 3)

| # | Question | Average | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|---------|--------------|-----------|--------|---------------|------------|
| 1 | Reranking Task 7 ưu tiên dùng API nào? | 0.252 | 0.901 | 0.008 | 0.000 | Retrieval/Generation | Expected context chưa khớp mạnh với chunks hoặc dữ liệu OCR còn thiếu. |
| 2 | Task 10 tạo câu trả lời bằng cách nào? | 0.260 | 0.911 | 0.042 | 0.000 | Retrieval/Generation | Expected context chưa khớp mạnh với chunks hoặc dữ liệu OCR còn thiếu. |
| 3 | Vector store trong Task 4 ưu tiên dịch vụ nào khi có key? | 0.279 | 0.940 | 0.021 | 0.000 | Retrieval/Generation | Expected context chưa khớp mạnh với chunks hoặc dữ liệu OCR còn thiếu. |

## Recommendations

### Cải tiến 1
**Action:** OCR lại đầy đủ các PDF pháp luật để thay các placeholder markdown.
**Expected impact:** Tăng context recall và faithfulness cho câu hỏi pháp luật chi tiết.

### Cải tiến 2
**Action:** Tune score threshold và ưu tiên PageIndex fallback cho câu hỏi pháp luật dài.
**Expected impact:** Truy xuất đúng đoạn trong PDF hơn khi hybrid local index yếu.

### Cải tiến 3
**Action:** Khi demo có mạng ổn định, bật Gemini/Jina thay cho fallback local.
**Expected impact:** Câu trả lời tự nhiên hơn và reranking chính xác hơn.

## Run Metadata

- Golden dataset size: 15
- Configs compared: 2
- Metrics: 4 required metrics + average
