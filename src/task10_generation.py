"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    # TODO: Implement reordering
    #
    # if len(chunks) <= 2:
    #     return chunks
    #
    # # Split into first half (important → đầu) and second half (important → cuối)
    # reordered = []
    # for i in range(0, len(chunks), 2):
    #     reordered.append(chunks[i])  # Odd positions go first
    # for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
    #     reordered.append(chunks[i])  # Even positions go last (reversed)
    #
    # return reordered
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    last_even_index = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(last_even_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    # TODO: Implement context formatting
    #
    # context_parts = []
    # for i, chunk in enumerate(chunks, 1):
    #     source = chunk.get("metadata", {}).get("source", f"Source {i}")
    #     doc_type = chunk.get("metadata", {}).get("type", "unknown")
    #     context_parts.append(
    #         f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
    #         f"{chunk['content']}\n"
    #     )
    # return "\n---\n".join(context_parts)
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"source_{i}")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _generate_local_answer(query: str, chunks: list[dict]) -> str:
    """Create a deterministic citation answer when no LLM API key is available."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = [
        "Dựa trên các nguồn đã truy xuất, tôi có thể tóm tắt như sau:"
    ]

    for chunk in chunks[:3]:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Nguồn không rõ")
        content = chunk.get("content", "").strip().replace("\n", " ")
        if not content:
            continue
        snippet = content[:260].rstrip()
        if len(content) > 260:
            snippet += "..."
        answer_parts.append(f"- {snippet} [{source}]")

    if len(answer_parts) == 1:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return "\n".join(answer_parts)


def _generate_with_gemini_api(query: str, context: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ""

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    response = requests.post(
        endpoint,
        params={"key": api_key},
        json={
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "topP": TOP_P,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    parts = (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    return "\n".join(part.get("text", "") for part in parts).strip()


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # TODO: Implement generation pipeline
    #
    # # Step 1: Retrieve
    # chunks = retrieve(query, top_k=top_k)
    #
    # # Step 2: Reorder
    # reordered = reorder_for_llm(chunks)
    #
    # # Step 3: Format context
    # context = format_context(reordered)
    #
    # # Step 4: Build prompt
    # user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
    #
    # # Step 5: Call LLM
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user_message}
    #     ],
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    # )
    #
    # answer = response.choices[0].message.content
    #
    # # Step 6: Return
    # return {
    #     "answer": answer,
    #     "sources": chunks,
    #     "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    # }
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "context": context,
            "generation_method": "none",
        }

    generation_method = "local_fallback"
    try:
        answer = _generate_with_gemini_api(query, context)
        if answer:
            generation_method = "gemini_api"
        else:
            answer = _generate_local_answer(query, reordered)
    except Exception:
        answer = _generate_local_answer(query, reordered)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if generation_method == "local_fallback" and api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
            generation_method = "openai_api"
        except Exception:
            answer = _generate_local_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "context": context,
        "generation_method": generation_method,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
