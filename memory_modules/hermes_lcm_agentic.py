"""Store-backed Codex agent memory for the H6 Hermes-LCM agentic lane."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .codex import (
    DEFAULT_CODEX_BINARY,
    DEFAULT_CODEX_MAX_ATTEMPTS,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    DEFAULT_CODEX_TIMEOUT_SECONDS,
    PROCESS_POLL_INTERVAL_SECONDS,
    copy_question_image,
    ensure_string_list,
    load_json,
    load_question_index,
    parse_codex_json_events,
    question_image,
    read_memory_output_status,
    save_json,
    terminate_process_group,
    utc_now_iso,
    validate_memory_module_output_payload,
)
from .hermes_search import (
    connect_read_only,
    load_span_rows,
    resolve_scope_sources,
)
from .memory import Memory, MemoryConfig, MemoryContextItem, register_memory, require


DEFAULT_PROMPT = (
    "You are acting as a memory retrieval module. "
    "Read INSTRUCTION.md and question.json in the current directory. "
    "Use the local hermes_search.py CLI to explore only the current question's "
    "read-only Hermes-LCM trajectory scope. "
    "Write the final valid JSON result to memory_module_output.json, overwriting "
    "any previous copy."
)
QUESTION_INSTRUCTIONS = """# Instructions

You are acting as a memory retrieval module, not as the final answering model.

Read `question.json`. The canonical Hermes-LCM store is available only through
the local read-only `hermes_search.py` CLI. Its generated `hermes_scope.json`
limits every lexical, semantic, fusion, and adjacency candidate to this
question's `haystack_ids`. Do not access or modify the canonical database
directly.

Search examples:

```bash
{{HERMES_SEARCH_PYTHON}} hermes_search.py search --query "distinctive terms from the question"
{{HERMES_SEARCH_PYTHON}} hermes_search.py search --query '"exact phrase" alternative terms' --limit 20
{{HERMES_SEARCH_PYTHON}} hermes_search.py show --trajectory-id TRAJECTORY_ID --start-state 3 --end-state 5
```

Important rules:

- Use the search CLI before returning your result. Try focused query rewrites
  when the first search is weak.
- Do not answer the benchmark question directly.
- Put the most important evidence first and avoid redundant spans.
- Use only trajectory and zero-based state ids printed by the CLI.
- The total number of states across all spans must be at most 20. Span bounds
  are inclusive.
- Do not copy screenshots or large AXTree blocks into the output JSON.

Write `memory_module_output.json` as valid JSON with exactly this schema:

```json
{
  "memory_markdown": "## Support Analysis\\n...\\n\\n## Relevant Procedure and Hint Notes\\n...",
  "trajectory_spans": [
    {
      "trajectory_id": "<trajectory id>",
      "start_state_index": 0,
      "end_state_index": 0
    }
  ]
}
```

`memory_markdown` contains the two narrative sections above. It may mention the
likely answer when strongly supported. If no useful evidence exists, still
write valid JSON with minimal markdown and an empty `trajectory_spans` list.
"""


def _required_text(params: dict[str, object], key: str) -> str:
    value = params.get(key)
    require(
        isinstance(value, str) and value.strip(),
        f"hermes_lcm_agentic {key} must be a non-empty string",
    )
    return value.strip()


def _bounded_int(
    params: dict[str, object],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = params.get(key, default)
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"hermes_lcm_agentic {key} must be an integer",
    )
    require(
        minimum <= value <= maximum,
        f"hermes_lcm_agentic {key} must be between {minimum} and {maximum}",
    )
    return value


def _bounded_float(
    params: dict[str, object],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = params.get(key, default)
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"hermes_lcm_agentic {key} must be a number",
    )
    number = float(value)
    require(
        minimum <= number <= maximum,
        f"hermes_lcm_agentic {key} must be between {minimum} and {maximum}",
    )
    return number


@register_memory
class HermesLCMAgenticMemory(Memory):
    """Codex agent over a scoped, canonical, read-only Hermes-LCM store."""

    memory_type = "hermes_lcm_agentic"

    _PERSISTED_PARAMS = frozenset(
        {
            "questions_path",
            "evidence_mode",
            "canonical_store_path",
            "asset_root",
            "product_root",
            "retrieval_params",
            "codex_params",
        }
    )
    _RUNTIME_PARAMS = frozenset(
        {
            "workspace_dir",
            "trajectories_root_dir",
            "query_trace_dir",
        }
    )

    def __init__(self, memory_params: dict[str, object]) -> None:
        unexpected = sorted(
            set(memory_params) - self._PERSISTED_PARAMS - self._RUNTIME_PARAMS
        )
        require(
            not unexpected,
            f"hermes_lcm_agentic memory_params contains unexpected keys: {unexpected}",
        )
        questions_path = Path(_required_text(memory_params, "questions_path")).resolve()
        canonical_store_path = Path(
            _required_text(memory_params, "canonical_store_path")
        ).resolve()
        asset_root = Path(_required_text(memory_params, "asset_root")).resolve()
        product_root = Path(_required_text(memory_params, "product_root")).resolve()
        evidence_mode = memory_params.get("evidence_mode", "both")
        require(
            evidence_mode in {"axtree", "image", "both"},
            "hermes_lcm_agentic evidence_mode must be one of: axtree, image, both",
        )
        require(questions_path.is_file(), f"Missing questions file: {questions_path}")
        require(
            canonical_store_path.is_file(),
            f"Missing canonical Hermes-LCM store: {canonical_store_path}",
        )
        require(asset_root.is_dir(), f"Missing trajectory asset root: {asset_root}")
        require(
            (product_root / "__init__.py").is_file(),
            f"Product root is not a Hermes-LCM checkout: {product_root}",
        )

        retrieval_obj = memory_params.get("retrieval_params", {})
        codex_obj = memory_params.get("codex_params", {})
        require(
            isinstance(retrieval_obj, dict),
            "hermes_lcm_agentic retrieval_params must be an object",
        )
        require(
            isinstance(codex_obj, dict),
            "hermes_lcm_agentic codex_params must be an object",
        )
        retrieval_params = dict(retrieval_obj)
        codex_params = dict(codex_obj)
        semantic_enabled = retrieval_params.get("semantic_enabled", True)
        include_adjacent = retrieval_params.get("include_adjacent", True)
        require(
            isinstance(semantic_enabled, bool),
            "hermes_lcm_agentic retrieval_params.semantic_enabled must be a boolean",
        )
        require(
            isinstance(include_adjacent, bool),
            "hermes_lcm_agentic retrieval_params.include_adjacent must be a boolean",
        )
        semantic_provider = str(
            retrieval_params.get("semantic_provider", "") or ""
        ).strip()
        semantic_model = str(retrieval_params.get("semantic_model", "") or "").strip()
        if semantic_enabled:
            require(
                semantic_provider and semantic_model,
                "hermes_lcm_agentic semantic provider/model must be set when enabled",
            )
        normalized_retrieval: dict[str, object] = {
            "candidate_limit": _bounded_int(
                retrieval_params,
                "candidate_limit",
                default=128,
                minimum=1,
                maximum=128,
            ),
            "limit": _bounded_int(
                retrieval_params,
                "limit",
                default=16,
                minimum=1,
                maximum=24,
            ),
            "text_char_limit": _bounded_int(
                retrieval_params,
                "text_char_limit",
                default=2_000,
                minimum=256,
                maximum=8_000,
            ),
            "include_adjacent": include_adjacent,
            "semantic_enabled": semantic_enabled,
            "semantic_provider": semantic_provider,
            "semantic_model": semantic_model,
            "semantic_top_trajectories": _bounded_int(
                retrieval_params,
                "semantic_top_trajectories",
                default=12,
                minimum=1,
                maximum=32,
            ),
            "semantic_query_timeout_seconds": _bounded_float(
                retrieval_params,
                "semantic_query_timeout_seconds",
                default=5.0,
                minimum=0.1,
                maximum=30.0,
            ),
        }

        binary_value = str(codex_params.get("binary", DEFAULT_CODEX_BINARY)).strip()
        model = str(codex_params.get("model", DEFAULT_CODEX_MODEL)).strip()
        effort = str(
            codex_params.get("reasoning_effort", DEFAULT_CODEX_REASONING_EFFORT)
        ).strip()
        timeout_seconds = codex_params.get(
            "timeout_seconds",
            DEFAULT_CODEX_TIMEOUT_SECONDS,
        )
        max_attempts = codex_params.get(
            "max_attempts",
            codex_params.get("max_retries", DEFAULT_CODEX_MAX_ATTEMPTS),
        )
        prompt = str(codex_params.get("prompt", DEFAULT_PROMPT)).strip()
        require_evidence_gate = codex_params.get("require_evidence_gate", False)
        require(binary_value, "hermes_lcm_agentic codex binary must be non-empty")
        require(model, "hermes_lcm_agentic codex model must be non-empty")
        require(effort, "hermes_lcm_agentic codex reasoning effort must be non-empty")
        require(
            isinstance(timeout_seconds, (int, float))
            and not isinstance(timeout_seconds, bool)
            and float(timeout_seconds) > 0.0,
            "hermes_lcm_agentic codex timeout must be positive",
        )
        require(
            isinstance(max_attempts, int)
            and not isinstance(max_attempts, bool)
            and max_attempts > 0,
            "hermes_lcm_agentic codex max attempts must be positive",
        )
        require(prompt, "hermes_lcm_agentic codex prompt must be non-empty")
        require(
            isinstance(require_evidence_gate, bool),
            "hermes_lcm_agentic require_evidence_gate must be a boolean",
        )
        resolved_binary = (
            shutil.which(binary_value) if os.sep not in binary_value else None
        )
        codex_binary = (
            Path(resolved_binary).resolve()
            if resolved_binary is not None
            else Path(binary_value).expanduser().resolve()
        )
        require(codex_binary.is_file(), f"Codex binary does not exist: {codex_binary}")
        extra_config = ensure_string_list(
            codex_params.get("extra_config", []),
            field_name="hermes_lcm_agentic codex_params.extra_config",
        )
        extra_args = ensure_string_list(
            codex_params.get("extra_args", []),
            field_name="hermes_lcm_agentic codex_params.extra_args",
        )

        persisted: dict[str, object] = {
            "questions_path": str(questions_path),
            "evidence_mode": str(evidence_mode),
            "canonical_store_path": str(canonical_store_path),
            "asset_root": str(asset_root),
            "product_root": str(product_root),
            "retrieval_params": normalized_retrieval,
            "codex_params": {
                "binary": str(codex_binary),
                "model": model,
                "reasoning_effort": effort,
                "timeout_seconds": float(timeout_seconds),
                "max_retries": int(max_attempts),
                "prompt": prompt,
                "require_evidence_gate": require_evidence_gate,
                "extra_config": extra_config,
                "extra_args": extra_args,
            },
        }
        super().__init__(persisted)
        self.questions_path = questions_path
        self.question_by_id, self.question_id_by_text = load_question_index(
            questions_path
        )
        self.evidence_mode = str(evidence_mode)
        self.canonical_store_path = canonical_store_path
        self.asset_root = asset_root
        self.product_root = product_root
        self.retrieval_params = normalized_retrieval
        self.codex_binary = codex_binary
        self.codex_model = model
        self.codex_reasoning_effort = effort
        self.codex_timeout_seconds = float(timeout_seconds)
        self.codex_max_attempts = int(max_attempts)
        self.codex_prompt = prompt
        self.require_evidence_gate = require_evidence_gate
        self.codex_extra_config = extra_config
        self.codex_extra_args = extra_args
        self.workspace_dir = self._runtime_path(memory_params.get("workspace_dir"))
        self.trajectories_root_dir = self._runtime_path(
            memory_params.get("trajectories_root_dir")
        )
        self.query_trace_dir = self._runtime_path(memory_params.get("query_trace_dir"))
        self.cancel_event: threading.Event | None = None
        self.inserted_trajectory_ids: list[str] = []
        self.inserted_trajectory_id_set: set[str] = set()
        self.inserted_state_counts: dict[str, int] = {}
        self._attempt_dir_lock = threading.Lock()
        self._last_query_metadata: dict[str, object] = {}
        if self.workspace_dir is not None:
            self._ensure_workspace_layout()
        if self.query_trace_dir is not None:
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)
        with connect_read_only(self.canonical_store_path) as connection:
            row = connection.execute(
                """
                SELECT corpus_uid, status
                FROM lcm_trajectory_corpora
                WHERE singleton = 1
                """
            ).fetchone()
            require(
                row is not None and row["status"] == "complete" and row["corpus_uid"],
                "Canonical Hermes-LCM corpus must be complete",
            )
            self.corpus_uid = str(row["corpus_uid"])

    @staticmethod
    def _runtime_path(value: object) -> Path | None:
        if isinstance(value, Path):
            return value.resolve()
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
        return None

    @property
    def memory_config(self) -> MemoryConfig:
        return {
            "memory_type": self.memory_type,
            "memory_params": dict(self.memory_params),
        }

    @classmethod
    def reconcile_loaded_memory_config(
        cls,
        saved_config: MemoryConfig,
        requested_config: MemoryConfig | None,
    ) -> MemoryConfig:
        require(
            saved_config.get("memory_type") == cls.memory_type,
            "saved hermes_lcm_agentic memory type mismatch",
        )
        if requested_config is None:
            return {
                "memory_type": cls.memory_type,
                "memory_params": dict(saved_config["memory_params"]),
            }
        require(
            requested_config.get("memory_type") == cls.memory_type,
            "requested hermes_lcm_agentic memory type mismatch",
        )
        saved_params = dict(saved_config["memory_params"])
        requested_params = dict(requested_config["memory_params"])
        require(
            set(saved_params) == set(requested_params),
            "hermes_lcm_agentic requested config keys must match saved config",
        )
        runtime_paths = {
            "questions_path",
            "canonical_store_path",
            "asset_root",
            "product_root",
        }
        require(
            {
                key: value
                for key, value in saved_params.items()
                if key not in runtime_paths
            }
            == {
                key: value
                for key, value in requested_params.items()
                if key not in runtime_paths
            },
            "hermes_lcm_agentic immutable retrieval and Codex parameters must match",
        )
        return {
            "memory_type": cls.memory_type,
            "memory_params": requested_params,
        }

    def configure_runtime(self, **kwargs: object) -> None:
        query_trace_dir = kwargs.get("query_trace_dir")
        if query_trace_dir is not None:
            path = self._runtime_path(query_trace_dir)
            require(path is not None, "query_trace_dir must be a non-empty path")
            self.query_trace_dir = path
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)
        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None:
            require(
                isinstance(cancel_event, threading.Event),
                "cancel_event must be a threading.Event",
            )
            self.cancel_event = cancel_event

    def _ensure_workspace_layout(self) -> None:
        require(self.workspace_dir is not None, "workspace_dir is not configured")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    def _write_index_files(self) -> None:
        require(self.workspace_dir is not None, "workspace_dir is not configured")
        save_json(
            self.workspace_dir / "index.json",
            {
                "memory_type": self.memory_type,
                "updated_at_utc": utc_now_iso(),
                "canonical_store_path": str(self.canonical_store_path),
                "corpus_uid": self.corpus_uid,
                "trajectory_count": len(self.inserted_trajectory_ids),
                "inserted_trajectory_ids": list(self.inserted_trajectory_ids),
                "state_counts": dict(self.inserted_state_counts),
            },
        )
        save_json(
            self.workspace_dir / "haystack_manifest.json",
            {
                "id": f"{self.memory_type}_haystack",
                "variant": "canonical_store_question_scope",
                "trajectory_ids": list(self.inserted_trajectory_ids),
                "metadata": {
                    "generated_at_utc": utc_now_iso(),
                    "memory_type": self.memory_type,
                    "trajectory_count": len(self.inserted_trajectory_ids),
                    "corpus_uid": self.corpus_uid,
                },
            },
        )

    def _record_insert_failure(self, trajectory_id: str, detail: str) -> None:
        if self.workspace_dir is None:
            return
        self._ensure_workspace_layout()
        record = {
            "recorded_at_utc": utc_now_iso(),
            "trajectory_id": trajectory_id,
            "canonical_store_path": str(self.canonical_store_path),
            "detail": detail,
        }
        with (self.workspace_dir / "insert_failures.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def insert(self, trajectory: dict[str, object]) -> None:
        require(self.workspace_dir is not None, "insert requires workspace_dir")
        trajectory_id = trajectory.get("id")
        require(
            isinstance(trajectory_id, str) and trajectory_id.strip(),
            "insert trajectory id must be a non-empty string",
        )
        trajectory_id = trajectory_id.strip()
        if trajectory_id in self.inserted_trajectory_id_set:
            return None
        with connect_read_only(self.canonical_store_path) as connection:
            try:
                source = resolve_scope_sources(connection, [trajectory_id])[trajectory_id]
            except Exception as exc:
                detail = str(exc)
                self._record_insert_failure(trajectory_id, detail)
                raise RuntimeError(
                    "Canonical Hermes-LCM scope validation failed for "
                    f"trajectory_id={trajectory_id}: {detail}"
                ) from exc
        self.inserted_trajectory_ids.append(trajectory_id)
        self.inserted_trajectory_id_set.add(trajectory_id)
        self.inserted_state_counts[trajectory_id] = int(source["state_count"])
        self._write_index_files()
        return None

    def _is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled():
            raise KeyboardInterrupt("hermes_lcm_agentic query cancelled")

    def _next_attempt_dir(self, question_id: str) -> tuple[int, Path]:
        require(self.query_trace_dir is not None, "query_trace_dir is not configured")
        with self._attempt_dir_lock:
            question_dir = self.query_trace_dir / question_id
            question_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(
                path
                for path in question_dir.iterdir()
                if path.is_dir() and path.name.startswith("attempt_")
            )
            attempt_index = len(existing) + 1
            attempt_dir = question_dir / f"attempt_{attempt_index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
        return attempt_index, attempt_dir

    def _build_question_payload(
        self,
        *,
        query_text: str,
        query_image: str | None,
        sandbox_dir: Path,
    ) -> dict[str, Any]:
        if query_image is None:
            return {"question": query_text}
        image_name = copy_question_image(query_image, sandbox_dir)
        return {"question": {"text": query_text, "image": image_name}}

    def _build_codex_command(
        self,
        *,
        sandbox_dir: Path,
        last_message_path: Path,
    ) -> list[str]:
        command = [
            str(self.codex_binary),
            "exec",
            "-C",
            str(sandbox_dir),
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            "-o",
            str(last_message_path),
            "-m",
            self.codex_model,
            "-c",
            f"model_reasoning_effort={json.dumps(self.codex_reasoning_effort)}",
        ]
        for item in self.codex_extra_config:
            command.extend(["-c", item])
        command.extend(self.codex_extra_args)
        command.append(self.codex_prompt)
        return command

    def _scope_payload(self, question_id: str) -> dict[str, object]:
        return {
            "schema": "hermes-lcm-agentic-question-scope-v1",
            "question_id": question_id,
            "canonical_store_path": str(self.canonical_store_path),
            "asset_root": str(self.asset_root),
            "product_root": str(self.product_root),
            "corpus_uid": self.corpus_uid,
            "trajectory_ids": list(self.inserted_trajectory_ids),
            **self.retrieval_params,
        }

    def _normalize_output_for_query(self, output_path: Path) -> dict[str, Any]:
        payload = validate_memory_module_output_payload(
            load_json(output_path),
            require_evidence_gate=self.require_evidence_gate,
        )
        normalized: dict[str, Any] = {
            "memory_markdown": payload["memory_markdown"],
            "trajectory_spans_raw": payload["trajectory_spans"],
            "trajectory_spans_valid": [],
            "trajectory_spans_invalid": [],
        }
        if "evidence_status" in payload:
            normalized.update(
                {
                    "evidence_status": payload["evidence_status"],
                    "evidence_status_reason": payload["evidence_status_reason"],
                    "answer_policy": payload["answer_policy"],
                }
            )
        with connect_read_only(self.canonical_store_path) as connection:
            for span in payload["trajectory_spans"]:
                if span["trajectory_id"] not in self.inserted_trajectory_id_set:
                    normalized["trajectory_spans_invalid"].append(
                        {**span, "reason": "trajectory_outside_question_scope"}
                    )
                    continue
                try:
                    load_span_rows(
                        connection,
                        trajectory_id=span["trajectory_id"],
                        start_state_index=span["start_state_index"],
                        end_state_index=span["end_state_index"],
                        allowed_trajectory_ids=self.inserted_trajectory_ids,
                    )
                except Exception as exc:
                    normalized["trajectory_spans_invalid"].append(
                        {**span, "reason": str(exc)}
                    )
                    continue
                normalized["trajectory_spans_valid"].append(span)
        return normalized

    def _asset_path_from_row(self, row: Any) -> Path | None:
        relative = row["relative_path"]
        expected_sha = row["asset_sha256"]
        if relative is None or expected_sha is None:
            return None
        candidate = (self.asset_root / str(relative)).resolve()
        try:
            candidate.relative_to(self.asset_root)
        except ValueError as exc:
            raise RuntimeError("Stored screenshot path escapes the asset root") from exc
        require(candidate.is_file(), f"Stored screenshot is missing: {candidate}")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        require(
            digest == str(expected_sha),
            f"Stored screenshot hash changed: {candidate}",
        )
        return candidate

    def _build_memory_context_from_output(
        self,
        normalized_output: dict[str, Any],
    ) -> list[MemoryContextItem]:
        items: list[MemoryContextItem] = []
        memory_markdown = normalized_output["memory_markdown"]
        valid_spans = normalized_output["trajectory_spans_valid"]
        if isinstance(memory_markdown, str) and memory_markdown.strip():
            items.append({"type": "text", "value": memory_markdown.strip() + "\n"})
        if valid_spans:
            span_lines = ["## Trajectory State Spans"]
            for span in valid_spans:
                span_lines.append(
                    f"- {span['trajectory_id']}: states "
                    f"{span['start_state_index']}-{span['end_state_index']}"
                )
            items.append({"type": "text", "value": "\n".join(span_lines) + "\n"})
            items.append({"type": "text", "value": "## Linked Evidence\n"})

        with connect_read_only(self.canonical_store_path) as connection:
            for span_index, span in enumerate(valid_spans, start=1):
                rows = load_span_rows(
                    connection,
                    trajectory_id=span["trajectory_id"],
                    start_state_index=span["start_state_index"],
                    end_state_index=span["end_state_index"],
                    allowed_trajectory_ids=self.inserted_trajectory_ids,
                )
                first = rows[0]
                items.append(
                    {
                        "type": "text",
                        "value": (
                            f"### Trajectory span {span_index}: "
                            f"{span['trajectory_id']} states "
                            f"{span['start_state_index']}-{span['end_state_index']}\n\n"
                            f"Goal\n- {first['goal']}\n\n"
                            f"Outcome\n- {first['outcome'] or '<unknown>'}\n\n"
                            "Linked state evidence\n"
                        ),
                    }
                )
                for row in rows:
                    action = row["incoming_action"] or "<none>"
                    lines = [
                        f"State {row['state_index']} (step {row['step']})",
                        f"- URL: {row['url']}",
                        f"- Incoming action: {action}",
                    ]
                    if row["thoughts"]:
                        lines.append(f"- Thought: {row['thoughts']}")
                    if self.evidence_mode in {"axtree", "both"}:
                        lines.extend(["- AXTree:", str(row["text"])])
                    items.append(
                        {"type": "text", "value": "\n".join(lines) + "\n"}
                    )
                    if self.evidence_mode in {"image", "both"}:
                        screenshot = self._asset_path_from_row(row)
                        if screenshot is not None:
                            items.append({"type": "image", "value": str(screenshot)})
        return items

    def _run_query_attempt(
        self,
        *,
        question_id: str,
        query_text: str,
        query_image: str | None,
    ) -> dict[str, Any]:
        attempt_index, attempt_dir = self._next_attempt_dir(question_id)
        sandbox_dir = attempt_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        question_payload = self._build_question_payload(
            query_text=query_text,
            query_image=query_image,
            sandbox_dir=sandbox_dir,
        )
        save_json(sandbox_dir / "question.json", question_payload)
        save_json(sandbox_dir / "hermes_scope.json", self._scope_payload(question_id))
        (sandbox_dir / "INSTRUCTION.md").write_text(
            QUESTION_INSTRUCTIONS.replace(
                "{{HERMES_SEARCH_PYTHON}}",
                str(Path(sys.executable).resolve()),
            ),
            encoding="utf-8",
        )
        shutil.copy2(
            Path(__file__).with_name("hermes_search.py"),
            sandbox_dir / "hermes_search.py",
        )
        output_path = sandbox_dir / "memory_module_output.json"
        last_message_path = attempt_dir / "last_message.txt"
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        events_path = attempt_dir / "events.json"
        summary_path = attempt_dir / "summary.json"
        command = self._build_codex_command(
            sandbox_dir=sandbox_dir,
            last_message_path=last_message_path,
        )

        started_at_ts = time.time()
        timed_out = False
        interrupted = False
        stdout_text = ""
        stderr_text = ""
        returncode: int | None = None
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name == "posix"),
            )
            while True:
                elapsed_seconds = time.time() - started_at_ts
                remaining_seconds = self.codex_timeout_seconds - elapsed_seconds
                if self._is_cancelled():
                    interrupted = True
                    stdout_text, stderr_text = terminate_process_group(
                        process,
                        reason="cancel_event",
                    )
                    break
                if remaining_seconds <= 0:
                    timed_out = True
                    stdout_text, stderr_text = terminate_process_group(
                        process,
                        reason="timeout",
                    )
                    break
                try:
                    stdout_text, stderr_text = process.communicate(
                        timeout=min(
                            PROCESS_POLL_INTERVAL_SECONDS,
                            remaining_seconds,
                        )
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
            returncode = process.returncode
        except KeyboardInterrupt:
            if process is not None:
                stdout_text, stderr_text = terminate_process_group(
                    process,
                    reason="keyboard_interrupt",
                )
                returncode = process.returncode
            raise

        duration_seconds = time.time() - started_at_ts
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        events, usage = parse_codex_json_events(stdout_text)
        if events:
            save_json(events_path, events)
        status = read_memory_output_status(
            output_path,
            require_evidence_gate=self.require_evidence_gate,
        )
        raw_output = output_path.read_text(encoding="utf-8") if output_path.exists() else None
        summary: dict[str, Any] = {
            "question_id": question_id,
            "attempt_index": attempt_index,
            "command": command,
            "started_at_utc": datetime.fromtimestamp(
                started_at_ts,
                timezone.utc,
            ).isoformat(),
            "completed_at_utc": utc_now_iso(),
            "duration_seconds": duration_seconds,
            "returncode": returncode,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "status_after": status.state,
            "status_after_detail": status.detail,
            "usage": usage,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path) if events else None,
            "last_message_path": str(last_message_path),
            "output_path": str(output_path),
            "agent_output_raw_text": raw_output,
            "question_has_image": query_image is not None,
            "question_payload": question_payload,
            "canonical_store_path": str(self.canonical_store_path),
            "scope_trajectory_count": len(self.inserted_trajectory_ids),
        }
        if interrupted:
            summary["status_after"] = "interrupted"
            summary["status_after_detail"] = "query cancelled before Codex completed"
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": "interrupted",
                "detail": summary["status_after_detail"],
                "memory_context": [],
            }
        if not status.is_finished:
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": status.state,
                "detail": status.detail,
                "memory_context": [],
            }
        try:
            normalized_output = self._normalize_output_for_query(output_path)
            memory_context = self._build_memory_context_from_output(normalized_output)
        except Exception as exc:
            summary["status_after"] = "internal_postprocess_error"
            summary["status_after_detail"] = str(exc)
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": "internal_postprocess_error",
                "detail": str(exc),
                "memory_context": [],
            }
        summary.update(
            {
                "memory_markdown": normalized_output["memory_markdown"],
                "trajectory_spans_raw": normalized_output["trajectory_spans_raw"],
                "trajectory_spans_valid": normalized_output[
                    "trajectory_spans_valid"
                ],
                "trajectory_spans_invalid": normalized_output[
                    "trajectory_spans_invalid"
                ],
                "memory_context_item_count": len(memory_context),
            }
        )
        if "evidence_status" in normalized_output:
            summary.update(
                {
                    "evidence_status": normalized_output["evidence_status"],
                    "evidence_status_reason": normalized_output[
                        "evidence_status_reason"
                    ],
                    "answer_policy": normalized_output["answer_policy"],
                }
            )
        save_json(summary_path, summary)
        self._last_query_metadata = {
            "backend": self.memory_type,
            "corpus_uid": self.corpus_uid,
            "question_id": question_id,
            "attempt_index": attempt_index,
            "agent_returncode": returncode,
            "agent_duration_seconds": duration_seconds,
            "output_valid": True,
            "trajectory_spans_valid": len(
                normalized_output["trajectory_spans_valid"]
            ),
            "trajectory_spans_invalid": len(
                normalized_output["trajectory_spans_invalid"]
            ),
        }
        return {
            "success": True,
            "status": status.state,
            "detail": status.detail,
            "memory_context": memory_context,
        }

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        self._raise_if_cancelled()
        require(
            isinstance(query, str) and query.strip(),
            "hermes_lcm_agentic query must be non-empty",
        )
        require(self.workspace_dir is not None, "query requires workspace_dir")
        require(
            self.inserted_trajectory_ids,
            "query requires at least one validated trajectory id",
        )
        if self.query_trace_dir is None:
            self.query_trace_dir = (self.workspace_dir / "query_traces").resolve()
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)
        context = self.get_query_context()
        question_id_value = context.get("question_id")
        if isinstance(question_id_value, str) and question_id_value.strip():
            question_id = question_id_value
        else:
            question_id = self.question_id_by_text.get(query)
        require(
            isinstance(question_id, str) and question_id in self.question_by_id,
            "hermes_lcm_agentic could not resolve question id",
        )
        question_item = self.question_by_id[question_id]
        effective_query_image = query_image
        if effective_query_image is None:
            effective_query_image = question_image(question_item.get("question"))

        last_status = "unknown_failure"
        last_detail: str | None = None
        for attempt_number in range(1, self.codex_max_attempts + 1):
            self._raise_if_cancelled()
            result = self._run_query_attempt(
                question_id=question_id,
                query_text=query,
                query_image=effective_query_image,
            )
            if result["status"] == "interrupted":
                raise KeyboardInterrupt(
                    f"hermes_lcm_agentic query interrupted for {question_id}"
                )
            if result["success"]:
                return result["memory_context"]
            last_status = result["status"]
            last_detail = result["detail"]
            print(
                (
                    "[hermes_lcm_agentic] query attempt failed "
                    f"question_id={question_id} "
                    f"attempt={attempt_number}/{self.codex_max_attempts} "
                    f"status={last_status} detail={last_detail or 'n/a'}"
                ),
                file=sys.stderr,
                flush=True,
            )
        print(
            (
                "[hermes_lcm_agentic] returning empty context after "
                f"{self.codex_max_attempts} failed attempts "
                f"question_id={question_id} last_status={last_status} "
                f"last_detail={last_detail or 'n/a'}"
            ),
            file=sys.stderr,
            flush=True,
        )
        return []

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        del query, query_image, memory_context
        return dict(self._last_query_metadata)

    def _save_backend(self, output_dir: Path) -> None:
        require(self.workspace_dir is not None, "memory has no active workspace")
        self._write_index_files()
        if self.workspace_dir.resolve() == output_dir.resolve():
            return None
        shutil.copy2(self.workspace_dir / "index.json", output_dir / "index.json")
        shutil.copy2(
            self.workspace_dir / "haystack_manifest.json",
            output_dir / "haystack_manifest.json",
        )
        return None

    def _load_backend(self, input_dir: Path) -> None:
        self.workspace_dir = input_dir.resolve()
        self._ensure_workspace_layout()
        index = load_json(self.workspace_dir / "index.json")
        require(isinstance(index, dict), "index.json must contain an object")
        inserted_ids = index.get("inserted_trajectory_ids")
        require(
            isinstance(inserted_ids, list)
            and all(isinstance(item, str) and item for item in inserted_ids),
            "index.json inserted_trajectory_ids must be a string list",
        )
        with connect_read_only(self.canonical_store_path) as connection:
            sources = resolve_scope_sources(connection, inserted_ids)
        self.inserted_trajectory_ids = list(inserted_ids)
        self.inserted_trajectory_id_set = set(inserted_ids)
        self.inserted_state_counts = {
            trajectory_id: int(sources[trajectory_id]["state_count"])
            for trajectory_id in inserted_ids
        }
        return None
