"""
Task 8 - PageIndex Vectorless RAG.

Uses the real PageIndex SDK when PAGEINDEX_API_KEY is configured:
1. Upload legal PDFs from data/landing/legal to PageIndex.
2. Cache PageIndex doc_id values in data/pageindex/documents.json.
3. Query PageIndex retrieval API.

If the API is unavailable, pageindex_search falls back to a local lexical search
so offline tests and the rest of the pipeline still work.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexAPIError

try:
    from src.task4_chunking_indexing import load_index
except ModuleNotFoundError:
    from task4_chunking_indexing import load_index

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
LEGAL_PDF_DIR = PROJECT_DIR / "data" / "landing" / "legal"
PAGEINDEX_DIR = PROJECT_DIR / "data" / "pageindex"
PAGEINDEX_MANIFEST_PATH = PAGEINDEX_DIR / "documents.json"
PAGEINDEX_POLL_SECONDS = 2
PAGEINDEX_MAX_WAIT_SECONDS = 120


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _load_manifest() -> list[dict]:
    if not PAGEINDEX_MANIFEST_PATH.exists():
        return []
    payload = json.loads(PAGEINDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload.get("documents", [])


def _save_manifest(documents: list[dict]) -> None:
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST_PATH.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _remote_documents(client) -> dict[str, dict]:
    response = client.list_documents(limit=100)
    documents = response.get("documents") or response.get("data") or []
    remote_by_name = {}
    for document in documents:
        name = document.get("name") or document.get("filename")
        doc_id = document.get("id") or document.get("doc_id")
        if name and doc_id:
            remote_by_name[name] = {
                "doc_id": doc_id,
                "filename": name,
                "path": f"data/landing/legal/{name}",
                "type": "legal",
                "status": document.get("status", ""),
                "uploaded_at": document.get("createdAt", ""),
            }
    return remote_by_name


def upload_documents() -> list[dict]:
    """Upload local legal PDFs to PageIndex and cache their doc_id values."""
    if not PAGEINDEX_API_KEY:
        raise ValueError("Missing PAGEINDEX_API_KEY in .env")

    from pageindex import PageIndexClient

    existing = _load_manifest()
    existing_by_path = {doc.get("path"): doc for doc in existing}
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    remote_by_name = _remote_documents(client)
    documents: list[dict] = []

    for pdf_file in sorted(LEGAL_PDF_DIR.rglob("*.pdf")):
        relative_path = str(pdf_file.relative_to(PROJECT_DIR)).replace("\\", "/")
        cached = existing_by_path.get(relative_path)
        if cached and cached.get("doc_id"):
            documents.append(cached)
            print(f"  Reusing PageIndex doc: {pdf_file.name}")
            continue

        remote = remote_by_name.get(pdf_file.name)
        if remote and remote.get("doc_id"):
            remote["path"] = relative_path
            documents.append(remote)
            _save_manifest(documents)
            print(f"  Found existing PageIndex doc: {pdf_file.name}")
            continue

        try:
            response = client.submit_document(str(pdf_file))
        except PageIndexAPIError as exc:
            if "LimitReached" in str(exc):
                print("  PageIndex upload limit reached; using cached documents.")
                break
            raise

        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_file.name}")

        document = {
            "doc_id": doc_id,
            "filename": pdf_file.name,
            "path": relative_path,
            "type": "legal",
            "uploaded_at": int(time.time()),
        }
        documents.append(document)
        _save_manifest(documents)
        print(f"  Uploaded to PageIndex: {pdf_file.name}")

    _save_manifest(documents)
    return documents


def _wait_until_ready(client, doc_id: str) -> bool:
    deadline = time.time() + PAGEINDEX_MAX_WAIT_SECONDS
    while time.time() < deadline:
        if client.is_retrieval_ready(doc_id):
            return True
        time.sleep(PAGEINDEX_POLL_SECONDS)
    return False


def _extract_retrieval_items(payload: dict) -> list[dict]:
    retrieved_nodes = payload.get("retrieved_nodes")
    if isinstance(retrieved_nodes, list):
        items = []
        for node in retrieved_nodes:
            if not isinstance(node, dict):
                continue
            relevant_groups = node.get("relevant_contents", [])
            for group in relevant_groups:
                if not isinstance(group, list):
                    continue
                for content_item in group:
                    if not isinstance(content_item, dict):
                        continue
                    item = dict(content_item)
                    item["title"] = node.get("title", "")
                    item["node_id"] = node.get("id", "")
                    item["node_metadata"] = node.get("metadata", [])
                    items.append(item)
        return items

    for key in ("results", "retrieval", "chunks", "contexts", "nodes"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    data = payload.get("data")
    if isinstance(data, dict):
        return _extract_retrieval_items(data)
    if isinstance(data, list):
        return data
    return []


def _item_text(item: dict) -> str:
    for key in ("relevant_content", "text", "content", "markdown", "page_content", "answer"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(item, ensure_ascii=False)


def _item_score(item: dict) -> float:
    for key in ("score", "relevance_score", "similarity"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0


def _search_pageindex_api(query: str, top_k: int) -> list[dict]:
    if not PAGEINDEX_API_KEY:
        return []

    documents = _load_manifest()
    if not documents:
        documents = upload_documents()
    if not documents:
        return []

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    results: list[dict] = []

    for document in documents:
        doc_id = document.get("doc_id")
        if not doc_id:
            continue
        if not _wait_until_ready(client, doc_id):
            continue

        query_response = client.submit_query(doc_id=doc_id, query=query, thinking=False)
        retrieval_id = query_response.get("retrieval_id") or query_response.get("id")
        if not retrieval_id:
            continue

        retrieval_payload = {}
        deadline = time.time() + PAGEINDEX_MAX_WAIT_SECONDS
        while time.time() < deadline:
            retrieval_payload = client.get_retrieval(retrieval_id)
            status = str(retrieval_payload.get("status", "")).lower()
            if status in {"completed", "done", "success", "ready"}:
                break
            if _extract_retrieval_items(retrieval_payload):
                break
            time.sleep(PAGEINDEX_POLL_SECONDS)

        for item in _extract_retrieval_items(retrieval_payload):
            if not isinstance(item, dict):
                continue

            results.append(
                {
                    "content": _item_text(item),
                    "score": _item_score(item),
                    "metadata": {
                        "source": document.get("filename", ""),
                        "path": document.get("path", ""),
                        "type": document.get("type", "legal"),
                        "doc_id": doc_id,
                        "retrieval_id": retrieval_id,
                        "provider": "pageindex",
                    },
                    "source": "pageindex",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _search_local_fallback(query: str, top_k: int) -> list[dict]:
    query_tokens = _tokenize(query)
    chunks = load_index()

    results = []
    for chunk in chunks:
        content = chunk.get("content", "")
        content_tokens = _tokenize(content)
        overlap = len(query_tokens & content_tokens)
        coverage = overlap / (len(query_tokens) or 1)
        density = overlap / (len(content_tokens) or 1)
        score = 0.8 * coverage + 0.2 * density

        if score <= 0:
            continue

        metadata = dict(chunk.get("metadata", {}))
        metadata["provider"] = "local_pageindex_fallback"
        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": metadata,
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search PageIndex with the real API when possible.

    Returns list items with content, score, metadata, and source="pageindex".
    """
    if top_k <= 0:
        return []

    try:
        api_results = _search_pageindex_api(query, top_k)
        if api_results:
            return api_results
    except Exception as exc:
        print(f"PageIndex API failed ({type(exc).__name__}); using local fallback.")

    return _search_local_fallback(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Set PAGEINDEX_API_KEY in .env")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
        for result in results:
            preview = result["content"][:100].encode("ascii", errors="replace").decode("ascii")
            print(f"[{result['score']:.3f}] {result['source']} {preview}...")
