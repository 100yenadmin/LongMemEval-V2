"""Official-interface fixtures for the Hermes-LCM trajectory adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest


PRODUCT_ROOT = Path(
    os.environ.get(
        "HERMES_LCM_PRODUCT_ROOT",
        "/Volumes/LEXAR/repos/hermes-lcm-v2-trajectory-adapter",
    )
).resolve()
os.environ.setdefault("HERMES_LCM_PRODUCT_ROOT", str(PRODUCT_ROOT))

from memory_modules.memory import (  # noqa: E402
    MEMORY_TYPES,
    build_memory,
    load_memory,
    save_memory,
)
from evaluation import run_eval  # noqa: E402
from memory_modules import hermes_lcm as hermes_adapter  # noqa: E402


class _FakeTrajectoryProvider:
    provider_id = "fake"
    model_id = "fake-trajectory-v1"
    dim = 2

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0
        self.last_usage_tokens = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        self.last_usage_tokens = len(texts)
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        self.last_usage_tokens = 1
        return [1.0, 0.0]


class _FailingTrajectoryProvider(_FakeTrajectoryProvider):
    def embed_documents(self, texts):
        raise RuntimeError("synthetic semantic build failure")


def _write_png(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png" + payload)


def _trajectory(data_root: Path, trajectory_id: str = "trajectory-a") -> dict[str, object]:
    screenshots = data_root / "screenshots"
    for index in range(3):
        _write_png(screenshots / f"{trajectory_id}-{index}.png", bytes([index]))
    return {
        "id": trajectory_id,
        "goal": "Export the quarterly report",
        "start_url": "https://example.test/reports",
        "outcome": "Export failed",
        "states": [
            {
                "url": "https://example.test/reports",
                "action": None,
                "thought": "Need to export the report.",
                "accessibility_tree": "Reports page with an Export button.",
                "screenshot": f"{trajectory_id}-0.png",
            },
            {
                "url": "https://example.test/reports/export",
                "action": "Click Export",
                "thought": "The export should start.",
                "accessibility_tree": "Export failed because storage quota is full.",
                "screenshot": f"{trajectory_id}-1.png",
            },
            {
                "url": "https://example.test/settings/storage",
                "action": "Open storage settings",
                "thought": "Check quota before retrying.",
                "accessibility_tree": "Delete an old export before retrying.",
                "screenshot": f"{trajectory_id}-2.png",
            },
        ],
    }


def _config(tmp_path: Path, data_root: Path) -> dict[str, object]:
    return {
        "memory_type": "hermes_lcm",
        "memory_params": {
            "workspace_root": str((tmp_path / "workspaces").resolve()),
            "trajectories_root_dir": str(data_root.resolve()),
            "dataset_name": "example/trajectory-benchmark",
            "dataset_revision": "dataset-rev-1",
            "harness_commit": "harness-commit-1",
            "tier": "small",
            "domain": "enterprise",
            "candidate_limit": 16,
            "max_text_items": 4,
            "max_text_chars_per_item": 2000,
            "max_image_items": 2,
            "include_adjacent": True,
            "protect_sensitive": True,
        },
    }


def _semantic_config(tmp_path: Path, data_root: Path) -> dict[str, object]:
    config = _config(tmp_path, data_root)
    config["memory_params"].update({
        "semantic_enabled": True,
        "semantic_provider": "fake",
        "semantic_model": "fake-trajectory-v1",
        "semantic_top_trajectories": 4,
        "semantic_build_timeout_seconds": 30.0,
        "semantic_query_timeout_seconds": 3.0,
    })
    return config


def _built_memory(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    memory = build_memory(_config(tmp_path, data_root))
    memory.insert(_trajectory(data_root))
    return data_root, memory


def test_backend_is_registered_and_builds_through_official_memory_api(tmp_path: Path):
    assert "hermes_lcm" in MEMORY_TYPES
    data_root, memory = _built_memory(tmp_path)
    try:
        assert memory.memory_type == "hermes_lcm"
        assert Path(memory.memory_params["trajectories_root_dir"]) == data_root.resolve()
    finally:
        memory.close()


def test_insert_and_query_return_bounded_exact_text_and_existing_images(tmp_path: Path):
    _data_root, memory = _built_memory(tmp_path)
    try:
        context = memory.query("Why did export fail and what should happen before retrying?")
        text_items = [item for item in context if item["type"] == "text"]
        image_items = [item for item in context if item["type"] == "image"]
        assert 1 <= len(text_items) <= 4
        assert len(image_items) <= 2
        assert any("trajectory://" in item["value"] for item in text_items)
        assert any("storage quota" in item["value"] for item in text_items)
        assert any("before retrying" in item["value"] for item in text_items)
        assert all(Path(item["value"]).is_file() for item in image_items)
    finally:
        memory.close()


def test_query_returns_bounded_verbatim_excerpt_with_stable_exact_ref(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _config(tmp_path, data_root)
    config["memory_params"]["max_text_chars_per_item"] = 512
    memory = build_memory(config)
    trajectory = _trajectory(data_root)
    full_text = "prefix " * 500 + "needle exact answer" + " suffix" * 500
    trajectory["states"][1]["accessibility_tree"] = full_text
    try:
        memory.insert(trajectory)
        context = memory.query("needle exact answer")
        rendered = "\n".join(
            item["value"] for item in context if item["type"] == "text"
        )
        assert "Visible state excerpt (offset " in rendered
        assert "needle exact answer" in rendered
        assert "trajectory://" in rendered
        assert full_text not in rendered
    finally:
        memory.close()


def test_destination_state_action_is_rendered_as_incoming_action(tmp_path: Path):
    _data_root, memory = _built_memory(tmp_path)
    try:
        context = memory.query("Click Export storage quota", query_image=None)
        rendered = "\n".join(item["value"] for item in context if item["type"] == "text")
        assert "Incoming action: Click Export" in rendered
        assert "Outgoing action" not in rendered
    finally:
        memory.close()


def test_poison_query_context_cannot_change_result_or_trace(tmp_path: Path):
    _data_root, memory = _built_memory(tmp_path)
    try:
        question = "Why did export fail?"
        memory.set_query_context(
            question_id="scored-id-a",
            question_type="secret-type",
            question_item={"answer": "gold-a", "eval_function": "judge-a"},
        )
        first = memory.query(question)
        first_metadata = memory.post_query_hook(
            query=question,
            query_image=None,
            memory_context=first,
        )
        memory.set_query_context(
            question_id="scored-id-b",
            question_type="different-type",
            question_item={"answer": "gold-b", "eval_function": "judge-b"},
        )
        second = memory.query(question)
        second_metadata = memory.post_query_hook(
            query=question,
            query_image=None,
            memory_context=second,
        )
        assert second == first
        assert second_metadata == first_metadata
        serialized = json.dumps(second_metadata, sort_keys=True)
        assert "scored-id" not in serialized
        assert "gold-" not in serialized
        assert "secret-type" not in serialized
    finally:
        memory.close()


def test_query_image_is_input_only_and_never_returned(tmp_path: Path):
    data_root, memory = _built_memory(tmp_path)
    question_image = data_root / "question.png"
    _write_png(question_image, b"question")
    try:
        context = memory.query("Why did export fail?", query_image=str(question_image))
        returned_paths = {
            Path(item["value"]).resolve()
            for item in context
            if item["type"] == "image"
        }
        assert question_image.resolve() not in returned_paths
    finally:
        memory.close()


def test_save_load_uses_database_backup_and_preserves_query_bytes(tmp_path: Path):
    _data_root, memory = _built_memory(tmp_path)
    save_dir = tmp_path / "saved"
    try:
        before = memory.query("storage quota retrying")
        save_memory(memory, save_dir)
    finally:
        memory.close()

    config_text = (save_dir / "memory_config.json").read_text(encoding="utf-8")
    assert "api_key" not in config_text.casefold()
    assert "token" not in config_text.casefold()
    assert (save_dir / "lcm.db").is_file()
    assert (save_dir / "corpus-manifest.json").is_file()

    restored = load_memory(save_dir)
    try:
        after = restored.query("storage quota retrying")
        assert after == before
        assert restored.read_only is True
    finally:
        restored.close()


def test_save_load_allows_only_runtime_paths_to_rebase(tmp_path: Path):
    data_root, memory = _built_memory(tmp_path)
    save_dir = tmp_path / "saved-rebase"
    try:
        before = memory.query("storage quota")
        before_metadata = memory.post_query_hook(
            query="storage quota",
            query_image=None,
            memory_context=before,
        )
        save_memory(memory, save_dir)
    finally:
        memory.close()

    rebased_data = tmp_path / "rebased-data"
    shutil.copytree(data_root, rebased_data)
    requested = _config(tmp_path / "rebased-run", rebased_data)
    restored = load_memory(save_dir, requested_config=requested)
    try:
        after = restored.query("storage quota")
        after_metadata = restored.post_query_hook(
            query="storage quota",
            query_image=None,
            memory_context=after,
        )
        assert [item for item in after if item["type"] == "text"] == [
            item for item in before if item["type"] == "text"
        ]
        before_images = [
            Path(item["value"]).read_bytes()
            for item in before
            if item["type"] == "image"
        ]
        after_images = [
            Path(item["value"]).read_bytes()
            for item in after
            if item["type"] == "image"
        ]
        assert after_images == before_images
        assert after_metadata == before_metadata
        assert Path(restored.memory_params["workspace_root"]).is_relative_to(
            (tmp_path / "rebased-run").resolve()
        )
    finally:
        restored.close()

    incompatible = _config(tmp_path / "other-run", rebased_data)
    incompatible["memory_params"]["candidate_limit"] = 15
    with pytest.raises(RuntimeError, match="immutable corpus and retrieval"):
        load_memory(save_dir, requested_config=incompatible)


def test_explicit_product_root_wins_over_preloaded_installed_package(monkeypatch):
    fake_installed = ModuleType("hermes_lcm")
    fake_installed.__path__ = ["/definitely/not/the/candidate"]
    monkeypatch.setitem(sys.modules, "hermes_lcm", fake_installed)
    monkeypatch.setenv("HERMES_LCM_PRODUCT_ROOT", str(PRODUCT_ROOT))
    module = hermes_adapter._product_api()
    assert Path(module.__file__).resolve().is_relative_to(PRODUCT_ROOT)
    assert module.__name__.startswith("_hermes_lcm_eval_")


def test_config_rejects_credentials_and_requires_single_worker_policy(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _config(tmp_path, data_root)
    config["memory_params"]["api_key"] = "must-not-persist"
    with pytest.raises(RuntimeError, match="unexpected"):
        build_memory(config)

    clean = _config(tmp_path, data_root)
    memory = build_memory(clean)
    try:
        assert memory.supports_parallel_query is False
    finally:
        memory.close()


def test_run_eval_builds_pinned_config_and_rejects_parallel_prompt_workers(
    tmp_path: Path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_root.mkdir()
    args = SimpleNamespace(
        method="hermes_lcm",
        dataset_revision="dataset-rev-1",
        tier="small",
        domain="web",
    )
    config = run_eval.build_memory_config(args, data_root, output_dir)
    params = config["memory_params"]
    assert config["memory_type"] == "hermes_lcm"
    assert params["dataset_revision"] == "dataset-rev-1"
    assert params["harness_commit"] == run_eval.OFFICIAL_HARNESS_COMMIT
    assert params["protect_sensitive"] is True
    assert params["max_text_chars_per_item"] == 2000
    assert params["semantic_enabled"] is False
    assert params["semantic_provider"] == ""
    assert params["semantic_model"] == ""

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation.run_eval",
            "--data-root",
            str(data_root),
            "--domain",
            "web",
            "--method",
            "hermes_lcm",
            "--output-dir",
            str(output_dir),
            "--dataset-revision",
            "dataset-rev-1",
            "--prompt-build-max-workers",
            "2",
        ],
    )
    with pytest.raises(SystemExit, match="prompt-build-max-workers 1"):
        run_eval.main()


def test_semantic_mode_builds_before_query_and_reports_provider_usage(
    tmp_path: Path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    memory = build_memory(_semantic_config(tmp_path, data_root))
    provider = _FakeTrajectoryProvider()
    monkeypatch.setattr(
        memory._api,
        "create_trajectory_embedding_provider",
        lambda *_args, **_kwargs: provider,
    )
    try:
        memory.insert(_trajectory(data_root))
        context = memory.query("Why did export fail?")
        metadata = memory.post_query_hook(
            query="Why did export fail?",
            query_image=None,
            memory_context=context,
        )
        assert provider.document_calls == 1
        assert provider.query_calls == 1
        assert metadata["semantic_enabled"] is True
        assert metadata["semantic_provider"] == "fake"
        assert metadata["semantic_model"] == "fake-trajectory-v1"
        assert metadata["embedding_document_calls"] == 1
        assert metadata["embedding_query_calls"] == 1
        assert memory._store.manifest()["semantic_index"]["document_count"] == 1
    finally:
        memory.close()


def test_semantic_config_has_no_credentials_and_disabled_mode_stays_provider_free(
    tmp_path: Path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _config(tmp_path, data_root)
    memory = build_memory(config)
    monkeypatch.setattr(
        memory._api,
        "create_trajectory_embedding_provider",
        lambda *_args, **_kwargs: pytest.fail("disabled mode resolved a provider"),
    )
    try:
        memory.insert(_trajectory(data_root))
        context = memory.query("storage quota")
        metadata = memory.post_query_hook(
            query="storage quota",
            query_image=None,
            memory_context=context,
        )
        assert metadata["provider_calls"] == 0
        assert memory._store.manifest()["semantic_index"] is None
        assert not any("key" in key.casefold() or "token" in key.casefold()
                       for key in memory.memory_params)
    finally:
        memory.close()


def test_semantic_build_failure_falls_back_to_exact_fts_without_aborting(
    tmp_path: Path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    memory = build_memory(_semantic_config(tmp_path, data_root))
    monkeypatch.setattr(
        memory._api,
        "create_trajectory_embedding_provider",
        lambda *_args, **_kwargs: _FailingTrajectoryProvider(),
    )
    try:
        memory.insert(_trajectory(data_root))
        context = memory.query("storage quota")
        metadata = memory.post_query_hook(
            query="storage quota",
            query_image=None,
            memory_context=context,
        )
        rendered = "\n".join(
            item["value"] for item in context if item["type"] == "text"
        )
        assert "storage quota" in rendered
        assert metadata["semantic_setup_status"] == "fallback"
        assert metadata["semantic_setup_error"] == "RuntimeError"
        assert metadata["embedding_fallbacks"] == 1
    finally:
        memory.close()


def test_load_accepts_legacy_manifest_without_optional_semantic_index(
    tmp_path: Path,
):
    _data_root, memory = _built_memory(tmp_path)
    save_dir = tmp_path / "legacy-save"
    try:
        save_memory(memory, save_dir)
    finally:
        memory.close()
    manifest_path = save_dir / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.pop("semantic_index") is None
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restored = load_memory(save_dir)
    try:
        assert restored.query("storage quota")
    finally:
        restored.close()


def _build_semantic_memory(tmp_path: Path, monkeypatch, data_root: Path):
    memory = build_memory(_semantic_config(tmp_path, data_root))
    monkeypatch.setattr(
        memory._api,
        "create_trajectory_embedding_provider",
        lambda *_args, **_kwargs: _FakeTrajectoryProvider(),
    )
    memory.insert(_trajectory(data_root))
    return memory


def test_semantic_telemetry_is_side_channel_and_rendering_byte_identical(
    tmp_path: Path,
    monkeypatch,
):
    # Isolation contract (#124): enabling the query_trace_dir side channel must
    # not change the reader prompt bytes (rendered evidence) or the post-query
    # metadata for any query, while writing a per-question telemetry file.
    # Two fresh, identically-built memories keep the cumulative store counters
    # aligned per query position so any perturbation would show up.
    data_root = tmp_path / "data"
    data_root.mkdir()
    baseline = _build_semantic_memory(tmp_path / "a", monkeypatch, data_root)
    instrumented = _build_semantic_memory(tmp_path / "b", monkeypatch, data_root)
    trace_dir = tmp_path / "query_traces"
    instrumented.configure_runtime(query_trace_dir=trace_dir)
    questions = [
        "Why did export fail?",
        "storage quota retrying",
        "Open storage settings before retrying",
    ]
    try:
        for index, question in enumerate(questions):
            base_ctx = baseline.query(question)
            base_meta = baseline.post_query_hook(
                query=question, query_image=None, memory_context=base_ctx
            )

            qid = f"question-{index}"
            instrumented.set_query_context(
                question_id=qid, question_type="t", question_item={}
            )
            try:
                inst_ctx = instrumented.query(question)
                inst_meta = instrumented.post_query_hook(
                    query=question, query_image=None, memory_context=inst_ctx
                )
            finally:
                instrumented.clear_query_context()

            # Byte-identical rendered evidence AND unchanged post-query metadata.
            assert inst_ctx == base_ctx
            assert inst_meta == base_meta

            # Side-channel telemetry file written with the documented schema.
            trace_path = trace_dir / qid / "hermes_lcm_semantic_telemetry.json"
            assert trace_path.is_file()
            record = json.loads(trace_path.read_text(encoding="utf-8"))
            assert record["question_id"] == qid
            assert record["delivered_evidence_refs"] == [
                item["value"].split("[", 1)[-1].split("]", 1)[0]
                for item in inst_ctx
                if item["type"] == "text"
            ]
            # attempt/ranks present when the product exposes the instrument.
            if record["semantic_attempt"] is not None:
                assert record["semantic_attempt"]["outcome"] == "success"
                assert record["source_candidate_ranks"]
                assert record["state_candidate_pool"]
                counters = record["semantic_attempt_counters"]
                assert counters["successes"] >= 1
                assert counters["fallbacks"] == 0

        assert sorted(p.name for p in trace_dir.iterdir()) == [
            "question-0",
            "question-1",
            "question-2",
        ]
    finally:
        baseline.close()
        instrumented.close()


@pytest.mark.parametrize("mode", ["trace_dir_is_a_file", "trace_dir_is_read_only"])
def test_telemetry_write_failure_never_fails_the_question(
    tmp_path: Path,
    monkeypatch,
    mode: str,
):
    # A1 regression: _write_query_trace runs inside query() before it returns,
    # so an unwritable query_trace_dir must degrade to a counted, logged warning
    # -- the question must still return byte-identical evidence.
    data_root = tmp_path / "data"
    data_root.mkdir()
    baseline = _build_semantic_memory(tmp_path / "a", monkeypatch, data_root)
    victim = _build_semantic_memory(tmp_path / "b", monkeypatch, data_root)

    if mode == "trace_dir_is_a_file":
        bad = tmp_path / "trace_is_file"
        bad.write_text("not a directory", encoding="utf-8")
    else:
        bad = tmp_path / "trace_ro"
        bad.mkdir()
        os.chmod(bad, 0o500)  # read+execute, no write
    victim.configure_runtime(query_trace_dir=bad)

    questions = ["Why did export fail?", "storage quota", "Open storage settings"]
    try:
        for index, question in enumerate(questions):
            base_ctx = baseline.query(question)
            victim.set_query_context(
                question_id=f"q-{index}", question_type="t", question_item={}
            )
            try:
                victim_ctx = victim.query(question)  # must NOT raise
            finally:
                victim.clear_query_context()
            assert victim_ctx == base_ctx  # byte-identical evidence preserved

        summary = victim.run_summary()
        assert summary["telemetry_write_failures"] == len(questions)
    finally:
        if mode == "trace_dir_is_read_only":
            os.chmod(bad, 0o700)  # restore so tmp cleanup can remove it
        baseline.close()
        victim.close()
