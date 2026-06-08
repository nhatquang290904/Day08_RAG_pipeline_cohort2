"""Streamlit app for the group RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    _generate_local_answer,
    format_context,
    generate_with_citation,
    reorder_for_llm,
)


st.set_page_config(
    page_title="DrugLaw RAG Chatbot",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 1180px;}
    [data-testid="stSidebar"] {min-width: 300px;}
    .source-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.65rem;
        background: #fafafa;
    }
    .source-title {font-weight: 650; margin-bottom: 0.25rem;}
    .meta {font-size: 0.84rem; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Xin chào. Hãy hỏi về pháp luật ma túy hoặc các tin tức liên quan.",
                "sources": [],
                "generation_method": "welcome",
                "retrieval_source": "none",
            }
        ]


def reset_chat() -> None:
    st.session_state.messages = []
    init_state()


def build_follow_up_query(question: str) -> str:
    previous_user_turns = [
        msg["content"]
        for msg in st.session_state.messages
        if msg.get("role") == "user"
    ][-2:]
    if not previous_user_turns:
        return question
    history = "\n".join(f"- {turn}" for turn in previous_user_turns)
    return f"Câu hỏi trước:\n{history}\n\nCâu hỏi hiện tại: {question}"


def answer_question(question: str, top_k: int, fast_mode: bool) -> dict:
    expanded_query = build_follow_up_query(question)
    if fast_mode:
        sources = retrieve(expanded_query, top_k=top_k, use_reranking=True)
        ordered_sources = reorder_for_llm(sources)
        return {
            "answer": _generate_local_answer(question, ordered_sources),
            "sources": sources,
            "context": format_context(ordered_sources),
            "retrieval_source": sources[0].get("source", "none") if sources else "none",
            "generation_method": "local_demo",
        }
    return generate_with_citation(expanded_query, top_k=top_k)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("Chưa có source documents.")
        return

    for idx, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("source") or metadata.get("path") or f"Source {idx}"
        doc_type = metadata.get("type", "unknown")
        provider = metadata.get("provider", source.get("source", "hybrid"))
        score = source.get("score", 0.0)
        preview = source.get("content", "").replace("\n", " ").strip()
        if len(preview) > 360:
            preview = preview[:360].rstrip() + "..."

        st.markdown(
            f"""
            <div class="source-box">
              <div class="source-title">{idx}. {title}</div>
              <div class="meta">type={doc_type} | provider={provider} | score={score:.3f}</div>
              <div>{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


init_state()

with st.sidebar:
    st.title("DrugLaw RAG")
    top_k = st.slider("Top K sources", min_value=2, max_value=8, value=5)
    fast_mode = st.toggle("Fast local demo mode", value=True)
    st.caption("Tắt chế độ nhanh để gọi generation API trong Task 10 khi network/API key sẵn sàng.")
    st.button("Clear chat", on_click=reset_chat, use_container_width=True)

    st.divider()
    st.subheader("Pipeline")
    st.write("Task 9 Retrieval")
    st.write("Task 10 Generation")
    st.write("Sources + conversation memory")

st.title("RAG Chatbot Pháp Luật Ma Túy")
st.caption("Trả lời có citation, hiển thị source documents và hỗ trợ follow-up questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Source documents", expanded=False):
                render_sources(message["sources"])
            st.caption(
                f"retrieval={message.get('retrieval_source', 'unknown')} | "
                f"generation={message.get('generation_method', 'unknown')}"
            )

question = st.chat_input("Nhập câu hỏi...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            result = answer_question(question, top_k=top_k, fast_mode=fast_mode)
        st.markdown(result["answer"])
        with st.expander("Source documents", expanded=True):
            render_sources(result.get("sources", []))
        st.caption(
            f"retrieval={result.get('retrieval_source', 'unknown')} | "
            f"generation={result.get('generation_method', 'unknown')}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "generation_method": result.get("generation_method", "unknown"),
            "retrieval_source": result.get("retrieval_source", "unknown"),
        }
    )
