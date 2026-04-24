#!/usr/bin/env python3
"""Oracle persona retrieval tools.

Implements a local Qdrant hybrid index (dense + sparse) over persona corpora
stored under /home/hermes/data/oracle-sources.

Tools:
- oracle_persona_sync: build/update persona index
- oracle_persona_search: hybrid retrieve citations (RRF/DBSF fusion)
- oracle_citation_validate: strict quote-in-source validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("ORACLE_DATA_ROOT", "/home/hermes/data/oracle-sources"))
QDRANT_PATH = Path(os.getenv("ORACLE_QDRANT_PATH", str(DATA_ROOT / "qdrant")))
DENSE_MODEL_NAME = os.getenv("ORACLE_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_DIM = int(os.getenv("ORACLE_SPARSE_DIM", "65536"))


try:  # Optional dependency: tool check_fn will hide tools if unavailable.
    from qdrant_client import QdrantClient, models
    _HAS_QDRANT = True
except Exception:
    QdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    _HAS_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]
    _HAS_ST = False

_EMBEDDER = None


def _check_oracle_requirements() -> bool:
    return _HAS_QDRANT and _HAS_ST and DATA_ROOT.exists()


def _collection_name(persona_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", str(persona_id).strip().lower())
    return f"oracle_{clean or 'persona'}"


def _stable_point_id(source_id: str) -> int:
    h = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:16]
    return int(h, 16)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _sparse_encode(text: str):
    if models is None:
        raise RuntimeError("qdrant models unavailable")
    counts = Counter(_tokenize(text))
    if not counts:
        return models.SparseVector(indices=[], values=[])

    by_index: Dict[int, float] = {}
    for token, tf in counts.items():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % SPARSE_DIM
        by_index[idx] = by_index.get(idx, 0.0) + float(tf)

    pairs = sorted(by_index.items(), key=lambda kv: kv[0])
    indices = [idx for idx, _ in pairs]
    values = [val for _, val in pairs]
    return models.SparseVector(indices=indices, values=values)


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if not _HAS_ST or SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        _EMBEDDER = SentenceTransformer(DENSE_MODEL_NAME)
    return _EMBEDDER


def _get_client():
    if not _HAS_QDRANT or QdrantClient is None:
        raise RuntimeError("qdrant-client not installed")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def _close_client_safely(client: Any) -> None:
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if not callable(close_fn):
        return
    try:
        close_fn()
    except Exception as e:
        logger.warning("Failed closing Qdrant client cleanly: %s", e)


@contextmanager
def _qdrant_client_context() -> Iterator[Any]:
    """Open a local Qdrant client and always release the filesystem lock."""
    client = _get_client()
    try:
        yield client
    finally:
        _close_client_safely(client)


def _persona_sources_path(persona_id: str) -> Path:
    return DATA_ROOT / "personas" / persona_id / "SOURCES.json"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_persona_records(persona_id: str, max_docs: int | None = None) -> List[Dict[str, Any]]:
    src_path = _persona_sources_path(persona_id)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing persona registry: {src_path}")

    root = json.loads(src_path.read_text(encoding="utf-8"))
    sources = root.get("sources") if isinstance(root, dict) else None
    if not isinstance(sources, list):
        return []

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for s in sources:
        if not isinstance(s, dict):
            continue
        idx_file = s.get("index_file")
        if not idx_file:
            continue
        p = Path(str(idx_file))
        for row in _iter_jsonl(p):
            source_id = str(row.get("source_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not source_id or not text:
                continue
            if source_id in seen:
                continue
            seen.add(source_id)
            norm = {
                "source_id": source_id,
                "persona_id": str(row.get("persona_id") or persona_id),
                "source_type": str(row.get("source_type") or s.get("type") or "unknown"),
                "text": text,
                "canonical_url": row.get("canonical_url"),
                "datetime": row.get("datetime") or row.get("created_at"),
            }
            out.append(norm)
            if max_docs is not None and len(out) >= max_docs:
                return out
    return out


def _ensure_collection(client, collection_name: str, dense_dim: int, rebuild: bool = False) -> None:
    exists = client.collection_exists(collection_name)
    if exists and rebuild:
        client.delete_collection(collection_name)
        exists = False

    if exists:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=int(dense_dim), distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def _embed_texts(texts: List[str]) -> List[List[float]]:
    emb = _get_embedder()
    vecs = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def _upsert_records(client, collection_name: str, rows: List[Dict[str, Any]], batch_size: int = 128) -> None:
    if not rows:
        return

    texts = [r["text"] for r in rows]
    dense = _embed_texts(texts)

    points = []
    for r, d in zip(rows, dense):
        sid = r["source_id"]
        payload = {
            "source_id": sid,
            "persona_id": r.get("persona_id"),
            "source_type": r.get("source_type"),
            "canonical_url": r.get("canonical_url"),
            "datetime": r.get("datetime"),
            "text": r.get("text"),
        }
        points.append(
            models.PointStruct(
                id=_stable_point_id(sid),
                vector={
                    "dense": d,
                    "sparse": _sparse_encode(r.get("text") or ""),
                },
                payload=payload,
            )
        )

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i : i + batch_size], wait=True)


def _sync_persona(persona_id: str, rebuild: bool = False, max_docs: int | None = None) -> Dict[str, Any]:
    rows = _load_persona_records(persona_id, max_docs=max_docs)
    if not rows:
        return {
            "success": False,
            "error": f"No records found for persona '{persona_id}'",
            "persona_id": persona_id,
        }

    col = _collection_name(persona_id)

    with _qdrant_client_context() as client:
        probe = _embed_texts([rows[0]["text"]])[0]
        _ensure_collection(client, col, dense_dim=len(probe), rebuild=rebuild)

        _upsert_records(client, col, rows)

        info = client.get_collection(col)
        points_count = getattr(info, "points_count", None)

    return {
        "success": True,
        "persona_id": persona_id,
        "collection": col,
        "records_seen": len(rows),
        "points_count": points_count,
        "rebuild": rebuild,
    }


def _search_persona(persona_id: str, query: str, top_k: int, fusion: str) -> Dict[str, Any]:
    col = _collection_name(persona_id)

    with _qdrant_client_context() as client:
        if not client.collection_exists(col):
            return {
                "success": False,
                "error": f"Collection '{col}' does not exist. Run oracle_persona_sync first.",
                "persona_id": persona_id,
                "query": query,
            }

        q_dense = _embed_texts([query])[0]
        q_sparse = _sparse_encode(query)

        fusion = (fusion or "rrf").lower().strip()
        if fusion not in {"rrf", "dbsf"}:
            return {
                "success": False,
                "error": "fusion must be one of: rrf, dbsf",
                "persona_id": persona_id,
                "query": query,
            }

        result = client.query_points(
            collection_name=col,
            prefetch=[
                models.Prefetch(query=q_dense, using="dense", limit=max(20, top_k * 4)),
                models.Prefetch(query=q_sparse, using="sparse", limit=max(20, top_k * 4)),
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=top_k,
            with_payload=True,
        )

        points = getattr(result, "points", result)
        citations = []
        for rank, p in enumerate(points, start=1):
            payload = getattr(p, "payload", {}) or {}
            text = str(payload.get("text") or "")
            citations.append(
                {
                    "rank": rank,
                    "score": getattr(p, "score", None),
                    "source_id": payload.get("source_id"),
                    "source_type": payload.get("source_type"),
                    "datetime": payload.get("datetime"),
                    "canonical_url": payload.get("canonical_url"),
                    "quote": text[:500],
                    "text": text,
                }
            )

    return {
        "success": True,
        "persona_id": persona_id,
        "query": query,
        "fusion": fusion,
        "top_k": top_k,
        "citations": citations,
    }


def oracle_persona_sync_tool(args: Dict[str, Any]) -> str:
    if not _check_oracle_requirements():
        return tool_error(
            "Oracle requirements missing. Need qdrant-client + sentence-transformers and existing ORACLE_DATA_ROOT.",
            requirements={
                "has_qdrant_client": _HAS_QDRANT,
                "has_sentence_transformers": _HAS_ST,
                "data_root_exists": DATA_ROOT.exists(),
            },
        )

    persona_id = str(args.get("persona_id") or "dejaru22").strip().lower()
    rebuild = bool(args.get("rebuild", False))
    max_docs = args.get("max_docs")
    try:
        max_docs = int(max_docs) if max_docs is not None else None
    except Exception:
        max_docs = None

    try:
        res = _sync_persona(persona_id=persona_id, rebuild=rebuild, max_docs=max_docs)
        return tool_result(res)
    except Exception as e:
        logger.exception("oracle_persona_sync failed: %s", e)
        return tool_error(f"oracle_persona_sync failed: {e}")


def oracle_persona_search_tool(args: Dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")

    if not _check_oracle_requirements():
        return tool_error(
            "Oracle requirements missing. Need qdrant-client + sentence-transformers and existing ORACLE_DATA_ROOT.",
            requirements={
                "has_qdrant_client": _HAS_QDRANT,
                "has_sentence_transformers": _HAS_ST,
                "data_root_exists": DATA_ROOT.exists(),
            },
        )

    persona_id = str(args.get("persona_id") or "dejaru22").strip().lower()
    top_k = int(args.get("top_k", 8))
    top_k = max(1, min(top_k, 25))
    fusion = str(args.get("fusion") or "rrf")

    try:
        res = _search_persona(persona_id=persona_id, query=query, top_k=top_k, fusion=fusion)
        err = str(res.get("error") or "") if isinstance(res, dict) else ""
        if (not res.get("success")) and "does not exist" in err.lower():
            sync_res = _sync_persona(persona_id=persona_id, rebuild=False, max_docs=None)
            if not sync_res.get("success"):
                return tool_error(
                    f"oracle_persona_search failed: collection missing and auto-sync failed: {sync_res.get('error')}",
                    search_result=res,
                    sync_result=sync_res,
                )
            res = _search_persona(persona_id=persona_id, query=query, top_k=top_k, fusion=fusion)
        return tool_result(res)
    except Exception as e:
        logger.exception("oracle_persona_search failed: %s", e)
        return tool_error(f"oracle_persona_search failed: {e}")


def _source_text_map(persona_id: str) -> Dict[str, str]:
    rows = _load_persona_records(persona_id)
    return {str(r.get("source_id")): str(r.get("text") or "") for r in rows if r.get("source_id")}


def oracle_citation_validate_tool(args: Dict[str, Any]) -> str:
    persona_id = str(args.get("persona_id") or "dejaru22").strip().lower()
    claims = args.get("claims")
    if not isinstance(claims, list) or not claims:
        return tool_error("claims must be a non-empty array")

    try:
        source_map = _source_text_map(persona_id)
    except Exception as e:
        return tool_error(f"Failed loading persona sources: {e}")

    results = []
    for item in claims:
        if not isinstance(item, dict):
            results.append({"valid": False, "error": "claim entry is not an object"})
            continue

        claim = str(item.get("claim") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        quote = str(item.get("quote") or item.get("quoted_text") or "").strip()

        if not source_id:
            results.append({"claim": claim, "source_id": source_id, "quote": quote, "valid": False, "error": "source_id missing"})
            continue

        source_text = source_map.get(source_id)
        if source_text is None:
            results.append({"claim": claim, "source_id": source_id, "quote": quote, "valid": False, "error": "source_id not found"})
            continue

        if not quote:
            results.append({"claim": claim, "source_id": source_id, "quote": quote, "valid": False, "error": "quote missing"})
            continue

        valid = quote.lower() in source_text.lower()
        results.append({
            "claim": claim,
            "source_id": source_id,
            "quote": quote,
            "valid": valid,
            "error": None if valid else "quote not found in cited source text",
        })

    all_valid = all(bool(r.get("valid")) for r in results)
    return tool_result(
        success=all_valid,
        persona_id=persona_id,
        checked=len(results),
        valid=sum(1 for r in results if r.get("valid")),
        invalid=sum(1 for r in results if not r.get("valid")),
        results=results,
    )


# Backward-compatible direct-import aliases for scripts/execute_code users.
def oracle_persona_sync(args: Dict[str, Any]) -> str:
    return oracle_persona_sync_tool(args)


def oracle_persona_search(args: Dict[str, Any]) -> str:
    return oracle_persona_search_tool(args)


def oracle_citation_validate(args: Dict[str, Any]) -> str:
    return oracle_citation_validate_tool(args)


ORACLE_PERSONA_SYNC_SCHEMA = {
    "name": "oracle_persona_sync",
    "description": (
        "Build or refresh the local Qdrant hybrid index (dense+sparse) for an Oracle persona corpus. "
        "Run this before querying if the collection is missing or stale."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "persona_id": {"type": "string", "description": "Persona ID (default: dejaru22)"},
            "rebuild": {"type": "boolean", "description": "If true, drop and rebuild the collection", "default": False},
            "max_docs": {"type": "integer", "description": "Optional cap for indexing docs", "minimum": 1},
        },
        "required": [],
    },
}


ORACLE_PERSONA_SEARCH_SCHEMA = {
    "name": "oracle_persona_search",
    "description": (
        "Hybrid-retrieve persona sources from local Qdrant using dense+sparse vectors fused with RRF or DBSF. "
        "Returns ranked citations with source IDs and quote text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "persona_id": {"type": "string", "description": "Persona ID (default: dejaru22)"},
            "query": {"type": "string", "description": "Natural-language retrieval query"},
            "top_k": {"type": "integer", "description": "Number of citations to return", "default": 8, "minimum": 1, "maximum": 25},
            "fusion": {"type": "string", "enum": ["rrf", "dbsf"], "default": "rrf", "description": "Fusion strategy for combining dense+sparse retrieval"},
        },
        "required": ["query"],
    },
}


ORACLE_CITATION_VALIDATE_SCHEMA = {
    "name": "oracle_citation_validate",
    "description": (
        "Strictly validate that quoted evidence actually appears in each cited source for persona answers. "
        "Use before final response in Oracle mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "persona_id": {"type": "string", "description": "Persona ID (default: dejaru22)"},
            "claims": {
                "type": "array",
                "description": "Claims with cited source IDs and verbatim quotes",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["source_id", "quote"],
                },
                "minItems": 1,
            },
        },
        "required": ["claims"],
    },
}


registry.register(
    name="oracle_persona_sync",
    toolset="oracle",
    schema=ORACLE_PERSONA_SYNC_SCHEMA,
    handler=lambda args, **kw: oracle_persona_sync_tool(args),
    check_fn=_check_oracle_requirements,
    emoji="🧱",
)

registry.register(
    name="oracle_persona_search",
    toolset="oracle",
    schema=ORACLE_PERSONA_SEARCH_SCHEMA,
    handler=lambda args, **kw: oracle_persona_search_tool(args),
    check_fn=_check_oracle_requirements,
    emoji="🔎",
)

registry.register(
    name="oracle_citation_validate",
    toolset="oracle",
    schema=ORACLE_CITATION_VALIDATE_SCHEMA,
    handler=lambda args, **kw: oracle_citation_validate_tool(args),
    check_fn=lambda: DATA_ROOT.exists(),
    emoji="✅",
)
