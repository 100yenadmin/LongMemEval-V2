from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from evaluation.harness import (
    NONSHARED_PARALLEL_MEMORY_TYPES,
    inject_runtime_memory_params,
)
from evaluation.run_eval import METHODS, build_memory_config
from memory_modules.hermes_lcm_agentic import HermesLCMAgenticMemory
from memory_modules.memory import (
    MEMORY_TYPES,
    Memory,
    build_memory,
    load_memory,
    save_memory,
)


CORPUS_UID = "a" * 64


def _create_store(tmp_path: Path) -> tuple[Path, Path]:
    store_dir = tmp_path / "canonical" / "enterprise"
    store_dir.mkdir(parents=True)
    db_path = store_dir / "lcm.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE lcm_trajectory_corpora (
            singleton INTEGER PRIMARY KEY,
            corpus_uid TEXT,
            status TEXT NOT NULL
        );
        CREATE TABLE lcm_trajectory_sources (
            source_id INTEGER PRIMARY KEY,
            trajectory_id TEXT NOT NULL UNIQUE,
            ordinal INTEGER NOT NULL UNIQUE,
            source_json TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            goal TEXT NOT NULL,
            start_url TEXT NOT NULL,
            outcome TEXT,
            state_count INTEGER NOT NULL,
            inserted_at REAL NOT NULL
        );
        CREATE TABLE lcm_trajectory_states (
            state_id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            state_index INTEGER NOT NULL,
            sequence_ordinal INTEGER NOT NULL,
            step INTEGER NOT NULL,
            url TEXT NOT NULL,
            incoming_action TEXT,
            thoughts TEXT,
            text TEXT NOT NULL,
            search_text TEXT NOT NULL,
            observed_at REAL,
            observed_at_source TEXT,
            occurred_at REAL,
            occurred_at_source TEXT,
            ingested_at REAL NOT NULL
        );
        CREATE TABLE lcm_trajectory_assets (
            state_id INTEGER PRIMARY KEY,
            relative_path TEXT,
            sha256 TEXT
        );
        CREATE VIRTUAL TABLE lcm_trajectory_states_fts
        USING fts5(search_text, content='lcm_trajectory_states', content_rowid='state_id');
        CREATE TABLE lcm_trajectory_embedding_profiles (
            profile_digest TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dim INTEGER NOT NULL,
            active INTEGER NOT NULL
        );
        CREATE TABLE lcm_trajectory_embeddings (
            source_id INTEGER PRIMARY KEY,
            profile_digest TEXT NOT NULL,
            vector BLOB NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO lcm_trajectory_corpora(singleton, corpus_uid, status) "
        "VALUES (1, ?, 'complete')",
        (CORPUS_UID,),
    )
    sources = [
        (1, "traj-in", 0, "Find the alpha setting", "completed"),
        (2, "traj-out", 1, "Unrelated out-of-scope source", "completed"),
    ]
    for source_id, trajectory_id, ordinal, goal, outcome in sources:
        connection.execute(
            """
            INSERT INTO lcm_trajectory_sources(
                source_id, trajectory_id, ordinal, source_json, source_sha256,
                goal, start_url, outcome, state_count, inserted_at
            ) VALUES (?, ?, ?, '{}', 'sha', ?, 'https://example.test', ?, 1, 0)
            """,
            (source_id, trajectory_id, ordinal, goal, outcome),
        )
    states = [
        (
            11,
            1,
            "The alpha setting is enabled after opening Preferences.",
            "open preferences",
        ),
        (
            22,
            2,
            "alpha alpha alpha alpha alpha should never escape the scope",
            "out-of-scope",
        ),
    ]
    for state_id, source_id, text, action in states:
        connection.execute(
            """
            INSERT INTO lcm_trajectory_states(
                state_id, source_id, state_index, sequence_ordinal, step, url,
                incoming_action, thoughts, text, search_text, ingested_at
            ) VALUES (?, ?, 0, 0, 0, 'https://example.test', ?, NULL, ?, ?, 0)
            """,
            (state_id, source_id, action, text, text),
        )
        connection.execute(
            "INSERT INTO lcm_trajectory_states_fts(rowid, search_text) VALUES (?, ?)",
            (state_id, text),
        )
    connection.commit()
    connection.close()
    return db_path, store_dir


def _write_product_root(tmp_path: Path) -> Path:
    product_root = tmp_path / "product"
    product_root.mkdir()
    (product_root / "__init__.py").write_text("", encoding="utf-8")
    return product_root


def _write_questions(tmp_path: Path) -> Path:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Where is the alpha setting?",
                "question_type": "static-environment",
                "eval_function": "exact_match",
                "answer": "Preferences",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return questions_path


def _write_fake_codex(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-codex"
    binary.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys

sandbox = Path(sys.argv[sys.argv.index("-C") + 1])
probe = subprocess.run(
    [sys.executable, str(sandbox / "hermes_search.py"), "search",
     "--query", "alpha setting", "--json"],
    cwd=sandbox,
    text=True,
    capture_output=True,
    check=True,
)
hits = json.loads(probe.stdout)["hits"]
assert hits and {hit["trajectory_id"] for hit in hits} == {"traj-in"}
(sandbox / "memory_module_output.json").write_text(
    json.dumps({
        "memory_markdown": (
            "## Support Analysis\\nThe scoped store supports the setting.\\n\\n"
            "## Relevant Procedure and Hint Notes\\nOpen Preferences."
        ),
        "trajectory_spans": [{
            "trajectory_id": "traj-in",
            "start_state_index": 0,
            "end_state_index": 0,
        }],
    }) + "\\n",
    encoding="utf-8",
)
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}))
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def _config(tmp_path: Path) -> dict[str, object]:
    db_path, _store_dir = _create_store(tmp_path)
    questions_path = _write_questions(tmp_path)
    product_root = _write_product_root(tmp_path)
    binary = _write_fake_codex(tmp_path)
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    return {
        "memory_type": "hermes_lcm_agentic",
        "memory_params": {
            "questions_path": str(questions_path),
            "evidence_mode": "axtree",
            "canonical_store_path": str(db_path),
            "asset_root": str(asset_root),
            "product_root": str(product_root),
            "retrieval_params": {
                "candidate_limit": 128,
                "limit": 16,
                "text_char_limit": 2000,
                "include_adjacent": True,
                "semantic_enabled": False,
                "semantic_provider": "",
                "semantic_model": "",
                "semantic_top_trajectories": 12,
                "semantic_query_timeout_seconds": 5.0,
            },
            "codex_params": {
                "binary": str(binary),
                "model": "gpt-5.4-mini",
                "reasoning_effort": "medium",
                "timeout_seconds": 30.0,
                "max_retries": 1,
                "extra_config": [],
                "extra_args": [],
            },
            "workspace_dir": str(tmp_path / "workspace"),
            "trajectories_root_dir": str(tmp_path),
            "query_trace_dir": str(tmp_path / "traces"),
        },
    }


def test_memory_abc_registration_insert_query_and_output_contract(tmp_path: Path):
    config = _config(tmp_path)
    assert "hermes_lcm_agentic" in MEMORY_TYPES
    assert issubclass(HermesLCMAgenticMemory, Memory)
    assert not inspect.isabstract(HermesLCMAgenticMemory)

    memory = build_memory(config)
    assert memory.insert({"id": "traj-in"}) is None
    memory.set_query_context(question_id="q1")
    context = memory.query("Where is the alpha setting?")
    assert isinstance(context, list)
    assert all(item["type"] in {"text", "image"} for item in context)
    rendered = "\n".join(
        item["value"] for item in context if item["type"] == "text"
    )
    assert "Open Preferences" in rendered
    assert "The alpha setting is enabled" in rendered

    summary_path = tmp_path / "traces" / "q1" / "attempt_001" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status_after"] == "finished"
    assert summary["returncode"] == 0
    assert len(summary["trajectory_spans_valid"]) == 1
    assert summary["trajectory_spans_invalid"] == []


def test_insert_missing_trajectory_fails_loud_and_writes_artifact(tmp_path: Path):
    memory = build_memory(_config(tmp_path))
    with pytest.raises(RuntimeError, match="scope validation failed"):
        memory.insert({"id": "missing-trajectory"})
    failure_path = tmp_path / "workspace" / "insert_failures.jsonl"
    record = json.loads(failure_path.read_text(encoding="utf-8"))
    assert record["trajectory_id"] == "missing-trajectory"
    assert "missing scoped trajectory ids" in record["detail"]


def test_search_cli_enforces_scope_inside_fts_candidates(tmp_path: Path):
    config = _config(tmp_path)
    params = config["memory_params"]
    scope_path = tmp_path / "hermes_scope.json"
    scope_path.write_text(
        json.dumps(
            {
                "canonical_store_path": params["canonical_store_path"],
                "asset_root": params["asset_root"],
                "product_root": params["product_root"],
                "trajectory_ids": ["traj-in"],
                **params["retrieval_params"],
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "memory_modules" / "hermes_search.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scope",
            str(scope_path),
            "search",
            "--query",
            "alpha",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["semantic_status"] == "disabled"
    assert payload["hits"]
    assert {hit["trajectory_id"] for hit in payload["hits"]} == {"traj-in"}


def test_memory_config_save_load_round_trip(tmp_path: Path):
    config = _config(tmp_path)
    memory = build_memory(config)
    memory.insert({"id": "traj-in"})
    save_dir = tmp_path / "saved"
    save_memory(memory, save_dir)
    loaded = load_memory(save_dir)
    assert loaded.memory_config == memory.memory_config
    assert loaded.inserted_trajectory_ids == ["traj-in"]


def test_harness_and_run_eval_conformance_surfaces(tmp_path: Path):
    config = _config(tmp_path)
    assert "hermes_lcm_agentic" in METHODS
    assert "hermes_lcm_agentic" in NONSHARED_PARALLEL_MEMORY_TYPES
    injected = inject_runtime_memory_params(
        {
            "memory_type": "hermes_lcm_agentic",
            "memory_params": {
                key: value
                for key, value in config["memory_params"].items()
                if key
                not in {"workspace_dir", "trajectories_root_dir", "query_trace_dir"}
            },
        },
        workspace_dir=tmp_path / "injected-workspace",
        trajectories_path=str(tmp_path / "trajectories.jsonl"),
        query_trace_dir=tmp_path / "injected-traces",
    )
    assert injected["memory_params"]["workspace_dir"].endswith(
        "injected-workspace"
    )
    assert injected["memory_params"]["query_trace_dir"].endswith("injected-traces")

    store_dir = Path(config["memory_params"]["canonical_store_path"]).parent
    canonical_params = {
        "trajectories_root_dir": config["memory_params"]["asset_root"],
        "domain": "enterprise",
        "candidate_limit": 128,
        "max_text_items": 16,
        "max_text_chars_per_item": 2000,
        "include_adjacent": True,
        "semantic_enabled": False,
        "semantic_provider": "",
        "semantic_model": "",
        "semantic_top_trajectories": 12,
        "semantic_query_timeout_seconds": 5.0,
    }
    (store_dir / "memory_config.json").write_text(
        json.dumps(
            {"memory_type": "hermes_lcm", "memory_params": canonical_params}
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        method="hermes_lcm_agentic",
        domain="enterprise",
        hermes_agentic_store_root=str(store_dir.parent),
        hermes_agentic_product_root=config["memory_params"]["product_root"],
        codex_binary=config["memory_params"]["codex_params"]["binary"],
        codex_model="gpt-5.4-mini",
        codex_reasoning_effort="medium",
        codex_timeout_seconds=30.0,
        codex_max_retries=1,
    )
    built = build_memory_config(args, Path(config["memory_params"]["asset_root"]))
    assert built["memory_type"] == "hermes_lcm_agentic"
    assert (
        built["memory_params"]["canonical_store_path"]
        == config["memory_params"]["canonical_store_path"]
    )
