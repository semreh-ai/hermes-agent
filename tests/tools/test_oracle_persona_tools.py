import json
from pathlib import Path

import pytest

from tools import oracle_persona_tools as opt


def _seed_persona(tmp_path: Path) -> Path:
    root = tmp_path / "oracle-sources"
    (root / "personas" / "dejaru22").mkdir(parents=True, exist_ok=True)

    x_dir = root / "xitter" / "dejaru22" / "index"
    x_dir.mkdir(parents=True, exist_ok=True)
    x_file = x_dir / "posts.jsonl"
    x_rows = [
        {
            "source_id": "dejaru22:tweet:1",
            "persona_id": "dejaru22",
            "source_type": "x_post",
            "post_id": "1",
            "text": "Discipline beats mood every day.",
            "canonical_url": "https://x.com/DejaRu22/status/1",
        },
        {
            "source_id": "dejaru22:tweet:2",
            "persona_id": "dejaru22",
            "source_type": "x_post",
            "post_id": "2",
            "text": "Your environment programs your behavior.",
            "canonical_url": "https://x.com/DejaRu22/status/2",
        },
    ]
    x_file.write_text("".join(json.dumps(r) + "\n" for r in x_rows), encoding="utf-8")

    tg_dir = root / "telegram" / "rubisroundtable" / "index"
    tg_dir.mkdir(parents=True, exist_ok=True)
    tg_file = tg_dir / "messages.jsonl"
    tg_rows = [
        {
            "source_id": "telegram:rubisroundtable:msg:10",
            "persona_id": "dejaru22",
            "source_type": "telegram_channel_post",
            "post_id": "10",
            "text": "Program or be programmed.",
            "canonical_url": "https://t.me/rubisroundtable/10",
        }
    ]
    tg_file.write_text("".join(json.dumps(r) + "\n" for r in tg_rows), encoding="utf-8")

    sources = {
        "persona_id": "dejaru22",
        "sources": [
            {
                "type": "x_twitter",
                "index_file": str(x_file),
            },
            {
                "type": "telegram_channel",
                "index_file": str(tg_file),
            },
        ],
    }
    (root / "personas" / "dejaru22" / "SOURCES.json").write_text(
        json.dumps(sources), encoding="utf-8"
    )
    return root


def test_load_persona_records_from_sources_registry(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    rows = opt._load_persona_records("dejaru22")

    assert len(rows) == 3
    assert {r["source_id"] for r in rows} == {
        "dejaru22:tweet:1",
        "dejaru22:tweet:2",
        "telegram:rubisroundtable:msg:10",
    }


def test_oracle_citation_validate_tool_checks_quote_membership(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    result = json.loads(
        opt.oracle_citation_validate_tool(
            {
                "persona_id": "dejaru22",
                "claims": [
                    {
                        "claim": "Use discipline",
                        "source_id": "dejaru22:tweet:1",
                        "quote": "Discipline beats mood",
                    },
                    {
                        "claim": "Wrong quote",
                        "source_id": "dejaru22:tweet:2",
                        "quote": "This quote is not present",
                    },
                ],
            }
        )
    )

    assert result["success"] is False
    assert len(result["results"]) == 2
    assert result["results"][0]["valid"] is True
    assert result["results"][1]["valid"] is False


def test_oracle_persona_search_requires_query():
    result = json.loads(opt.oracle_persona_search_tool({"persona_id": "dejaru22"}))
    assert "error" in result
    assert "query" in result["error"].lower()


def test_qdrant_client_context_closes_client_on_success(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = DummyClient()
    monkeypatch.setattr(opt, "_get_client", lambda: client)

    with opt._qdrant_client_context() as yielded:
        assert yielded is client
        assert client.closed is False

    assert client.closed is True


def test_qdrant_client_context_closes_client_on_exception(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = DummyClient()
    monkeypatch.setattr(opt, "_get_client", lambda: client)

    with pytest.raises(RuntimeError, match="boom"):
        with opt._qdrant_client_context():
            raise RuntimeError("boom")

    assert client.closed is True


def test_search_persona_closes_client_after_query(monkeypatch):
    closed = {"value": False}

    class DummyClient:
        def collection_exists(self, name):
            return True

        def query_points(self, **kwargs):
            point = type(
                "Point",
                (),
                {
                    "score": 0.9,
                    "payload": {
                        "source_id": "dejaru22:tweet:1",
                        "source_type": "x_post",
                        "datetime": None,
                        "canonical_url": "https://x.com/DejaRu22/status/1",
                        "text": "Discipline beats mood every day.",
                    },
                },
            )
            return type("Result", (), {"points": [point()]})()

        def close(self):
            closed["value"] = True

    class DummyModels:
        class Prefetch:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FusionQuery:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    monkeypatch.setattr(opt, "_get_client", lambda: DummyClient())
    monkeypatch.setattr(opt, "_embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(opt, "_sparse_encode", lambda text: {"sparse": text})
    monkeypatch.setattr(opt, "models", DummyModels)

    result = opt._search_persona("dejaru22", "discipline", 1, "rrf")

    assert result["success"] is True
    assert result["citations"][0]["source_id"] == "dejaru22:tweet:1"
    assert closed["value"] is True


def test_oracle_persona_search_auto_syncs_missing_collection(monkeypatch):
    calls = {"search": 0, "sync": 0}

    def fake_search(**kwargs):
        calls["search"] += 1
        if calls["search"] == 1:
            return {
                "success": False,
                "error": "Collection 'oracle_dejaru22' does not exist. Run oracle_persona_sync first.",
                "persona_id": kwargs["persona_id"],
                "query": kwargs["query"],
            }
        return {
            "success": True,
            "persona_id": kwargs["persona_id"],
            "query": kwargs["query"],
            "fusion": kwargs["fusion"],
            "top_k": kwargs["top_k"],
            "citations": [],
        }

    def fake_sync(**kwargs):
        calls["sync"] += 1
        return {"success": True, "persona_id": kwargs["persona_id"]}

    monkeypatch.setattr(opt, "_check_oracle_requirements", lambda: True)
    monkeypatch.setattr(opt, "_search_persona", fake_search)
    monkeypatch.setattr(opt, "_sync_persona", fake_sync)

    result = json.loads(
        opt.oracle_persona_search_tool(
            {"persona_id": "dejaru22", "query": "discipline", "top_k": 3, "fusion": "rrf"}
        )
    )

    assert result["success"] is True
    assert calls == {"search": 2, "sync": 1}


def test_direct_import_aliases_match_tool_functions(monkeypatch):
    monkeypatch.setattr(opt, "oracle_persona_search_tool", lambda args: json.dumps({"ok": args["query"]}))
    monkeypatch.setattr(opt, "oracle_persona_sync_tool", lambda args: json.dumps({"sync": True}))
    monkeypatch.setattr(opt, "oracle_citation_validate_tool", lambda args: json.dumps({"valid": True}))

    assert json.loads(opt.oracle_persona_search({"query": "discipline"})) == {"ok": "discipline"}
    assert json.loads(opt.oracle_persona_sync({})) == {"sync": True}
    assert json.loads(opt.oracle_citation_validate({"claims": [{"source_id": "x", "quote": "y"}]})) == {"valid": True}
