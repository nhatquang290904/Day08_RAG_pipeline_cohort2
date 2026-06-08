"""
Task 4 - Chunking and indexing.

Choices:
- Chunking: RecursiveCharacterTextSplitter. It is robust for mixed Markdown
  sources because it tries paragraph, line, sentence, word, then character
  boundaries. This fits legal text and crawled news without requiring perfect
  heading structure.
- chunk_size=500, chunk_overlap=50. 500 characters keeps retrieval snippets
  focused for citation; 50 characters preserves continuity across boundaries.
- Embedding: local-hashing-embedding-v1, 384 dimensions. This deterministic
  local embedding avoids network/model downloads in class while still creating
  comparable vectors for semantic search. It can be swapped later for
  sentence-transformers/all-MiniLM-L6-v2, which also uses 384 dimensions.
- Vector store: Weaviate Cloud when WEAVIATE_URL and WEAVIATE_API_KEY are set,
  with a local JSON cache kept for tests and offline fallback.
"""

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
INDEX_PATH = INDEX_DIR / "chunks_index.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
OPENAI_EMBEDDING_BATCH_SIZE = 64

VECTOR_STORE = "weaviate_cloud"
LOCAL_VECTOR_STORE = "local_json"
WEAVIATE_COLLECTION = "DrugLawDocs"


def _get_weaviate_cloud_config() -> tuple[str, str]:
    """Return configured Weaviate Cloud URL and API key, if available."""
    weaviate_url = (
        os.getenv("WEAVIATE_URL", "").strip()
        or os.getenv("WEAVIATE_CLUSTER_URL", "").strip()
    )
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if weaviate_url and not weaviate_url.startswith(("http://", "https://")):
        weaviate_url = f"https://{weaviate_url}"
    return weaviate_url, weaviate_api_key


def is_weaviate_cloud_configured() -> bool:
    """Check whether Task 4 should upload vectors to Weaviate Cloud."""
    weaviate_url, weaviate_api_key = _get_weaviate_cloud_config()
    return bool(weaviate_url and weaviate_api_key)


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {"content": str, "metadata": dict}
    """
    documents: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents with RecursiveCharacterTextSplitter.

    Returns:
        List of {"content": str, "metadata": dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for doc_index, doc in enumerate(documents):
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _hashing_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = _tokenize(text)

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _openai_embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with the OpenAI Embeddings API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), OPENAI_EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + OPENAI_EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def embed_query(text: str) -> list[float]:
    """Embed one query using the same backend as Task 4 indexing."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _openai_embed_texts([text])[0]
        except Exception:
            pass
    return _hashing_embedding(text)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an OpenAI API embedding to every chunk when OPENAI_API_KEY is available.

    Returns:
        Chunks with an "embedding" key.
    """
    texts = [chunk["content"] for chunk in chunks]
    embedding_backend = "local-hashing-fallback"
    try:
        if os.getenv("OPENAI_API_KEY"):
            vectors = _openai_embed_texts(texts)
            embedding_backend = "openai"
        else:
            vectors = [_hashing_embedding(text) for text in texts]
    except Exception as exc:
        print(f"OpenAI embedding failed ({type(exc).__name__}); using local fallback.")
        vectors = [_hashing_embedding(text) for text in texts]

    embedded_chunks: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        embedded = dict(chunk)
        embedded["metadata"] = dict(chunk.get("metadata", {}))
        embedded["metadata"]["embedding_backend"] = embedding_backend
        embedded["embedding"] = vector
        embedded_chunks.append(embedded)

    return embedded_chunks


def _index_to_weaviate_cloud(chunks: list[dict]) -> bool:
    """Upload chunks to Weaviate Cloud when WEAVIATE_URL/API_KEY are configured."""
    weaviate_url, weaviate_api_key = _get_weaviate_cloud_config()
    if not weaviate_url or not weaviate_api_key:
        return False
    weaviate_url = weaviate_url.rstrip("/")

    headers = {
        "Authorization": f"Bearer {weaviate_api_key}",
        "Content-Type": "application/json",
    }

    schema_response = requests.get(
        f"{weaviate_url}/v1/schema/{WEAVIATE_COLLECTION}",
        headers=headers,
        timeout=30,
    )

    if schema_response.status_code == 404:
        create_response = requests.post(
            f"{weaviate_url}/v1/schema",
            headers=headers,
            json={
                "class": WEAVIATE_COLLECTION,
                "vectorizer": "none",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "source", "dataType": ["text"]},
                    {"name": "doc_type", "dataType": ["text"]},
                    {"name": "path", "dataType": ["text"]},
                    {"name": "doc_index", "dataType": ["int"]},
                    {"name": "chunk_index", "dataType": ["int"]},
                    {"name": "embedding_backend", "dataType": ["text"]},
                ],
            },
            timeout=30,
        )
        create_response.raise_for_status()
    else:
        schema_response.raise_for_status()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        stable_key = f"{metadata.get('path', '')}:{metadata.get('chunk_index', 0)}"
        object_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        object_response = requests.put(
            f"{weaviate_url}/v1/objects/{WEAVIATE_COLLECTION}/{object_uuid}",
            headers=headers,
            json={
                "class": WEAVIATE_COLLECTION,
                "properties": {
                    "content": chunk.get("content", ""),
                    "source": metadata.get("source", ""),
                    "doc_type": metadata.get("type", ""),
                    "path": metadata.get("path", ""),
                    "doc_index": int(metadata.get("doc_index", 0)),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "embedding_backend": metadata.get("embedding_backend", ""),
                },
                "vector": chunk.get("embedding", []),
            },
            timeout=30,
        )
        object_response.raise_for_status()

    return True


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """
    Persist chunks locally and upload to Weaviate Cloud when configured.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embedding_backend = "none"
    if chunks:
        embedding_backend = chunks[0].get("metadata", {}).get("embedding_backend", "unknown")

    weaviate_uploaded = False
    try:
        weaviate_uploaded = _index_to_weaviate_cloud(chunks)
    except Exception as exc:
        print(f"Weaviate Cloud indexing failed ({type(exc).__name__}); kept local JSON index.")

    payload = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_backend": embedding_backend,
            "vector_store": VECTOR_STORE if weaviate_uploaded else LOCAL_VECTOR_STORE,
            "local_cache": str(INDEX_PATH),
            "weaviate_configured": is_weaviate_cloud_configured(),
            "weaviate_uploaded": weaviate_uploaded,
            "weaviate_collection": WEAVIATE_COLLECTION if weaviate_uploaded else "",
        },
        "chunks": chunks,
    }
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return INDEX_PATH


def load_index() -> list[dict]:
    """Load the local JSON index, building it if needed."""
    if not INDEX_PATH.exists():
        run_pipeline()
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return payload.get("chunks", [])


def load_index_config() -> dict:
    """Load index metadata/config."""
    if not INDEX_PATH.exists():
        run_pipeline()
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return payload.get("config", {})


def run_pipeline() -> Path:
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE if is_weaviate_cloud_configured() else LOCAL_VECTOR_STORE}")
    print(f"  Weaviate Collection: {WEAVIATE_COLLECTION}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    if embedded_chunks:
        backend = embedded_chunks[0].get("metadata", {}).get("embedding_backend", "unknown")
        print(f"Embedding backend: {backend}")

    index_path = index_to_vectorstore(embedded_chunks)
    config = load_index_config()
    if config.get("weaviate_uploaded"):
        print(f"Indexed to Weaviate Cloud collection: {WEAVIATE_COLLECTION}")
    else:
        print(f"Indexed to local JSON fallback: {index_path}")
    return index_path


if __name__ == "__main__":
    run_pipeline()
