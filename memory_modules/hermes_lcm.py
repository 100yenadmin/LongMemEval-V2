"""Official LongMemEval-V2 adapter for Hermes-LCM trajectory memory."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from .memory import Memory, MemoryContextItem, register_memory, require
from .trajectory_store import prepare_trajectory_insert


def _product_api():
    """Load the installed product package or an explicit source checkout.

    The source-root environment value is a development/bootstrap seam only. It
    is never copied into memory configuration, traces, manifests, or context.
    """
    root_value = os.environ.get("HERMES_LCM_PRODUCT_ROOT", "").strip()
    if root_value:
        root = Path(root_value).expanduser().resolve()
        init_path = root / "__init__.py"
        require(init_path.is_file(), "HERMES_LCM_PRODUCT_ROOT is not a product checkout")
        alias = "_hermes_lcm_eval_" + hashlib.sha256(
            str(root).encode("utf-8")
        ).hexdigest()[:16]
        if alias not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                alias,
                str(init_path),
                submodule_search_locations=[str(root)],
            )
            require(
                spec is not None and spec.loader is not None,
                "could not construct Hermes-LCM package spec",
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            spec.loader.exec_module(module)
        return importlib.import_module(f"{alias}.trajectory_store")
    try:
        return importlib.import_module("hermes_lcm.trajectory_store")
    except ModuleNotFoundError as first_error:
        raise RuntimeError(
            "Hermes-LCM product package is unavailable; install the plugin or "
            "set HERMES_LCM_PRODUCT_ROOT for this isolated evaluation"
        ) from first_error


def _arm_quota_from_env() -> tuple[int, int] | None:
    """Optional D(8,3)-style per-arm quota override for the H3.1 composition
    seam (a product's ``TrajectoryStore.query(..., arm_quota=(q_lex, q_sem))``
    kwarg -- see session-notes/2026-07-23/hermes-benchprog-h1/artifacts/
    H3.1-design-options.md, Option D). Read directly from the environment,
    exactly like the query-path spend-guard's LCM_EMBEDDING_QUERY_SPEND_*
    overrides -- never persisted into memory_params/config, so a product
    checkout that predates the ``arm_quota`` kwarg (e.g. the h1-v2-instrument
    control) is simply never passed it and is unaffected. Unset/empty means
    "current bytes" (kwarg omitted from the query() call entirely, not
    passed as None -- older products don't accept the keyword at all).
    """
    raw = os.environ.get("HERMES_LCM_ARM_QUOTA", "").strip()
    if not raw:
        return None
    parts = raw.split(",")
    require(len(parts) == 2, f"HERMES_LCM_ARM_QUOTA must be 'q_lex,q_sem', got {raw!r}")
    try:
        q_lex, q_sem = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise RuntimeError(f"HERMES_LCM_ARM_QUOTA must be two integers, got {raw!r}") from exc
    return (q_lex, q_sem)


def _lexical_floor_from_env() -> int:
    """Optional Policy-A lexical-floor override for the H3.1 A+D hybrid seam (a
    product's ``TrajectoryStore.query(..., lexical_floor=K)`` kwarg -- reserves
    the top ``K`` pure-BM25 incumbents a nucleus slot, composed on top of the
    ``arm_quota`` round-robin). Read from the environment exactly like
    ``_arm_quota_from_env`` -- never persisted into memory_params/config, so a
    product checkout that predates the ``lexical_floor`` kwarg is simply never
    passed it. Unset/empty/``0`` means "current bytes" (kwarg omitted from the
    query() call entirely, not passed as 0 -- older products don't accept it).
    """
    raw = os.environ.get("HERMES_LCM_LEXICAL_FLOOR", "").strip()
    if not raw:
        return 0
    try:
        floor = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"HERMES_LCM_LEXICAL_FLOOR must be an integer, got {raw!r}") from exc
    require(floor >= 0, f"HERMES_LCM_LEXICAL_FLOOR must be >= 0, got {floor}")
    return floor


def _required_text(params: dict[str, object], key: str) -> str:
    value = params.get(key)
    require(isinstance(value, str) and value.strip(), f"hermes_lcm {key} must be a non-empty string")
    return value.strip()


def _required_bool(params: dict[str, object], key: str) -> bool:
    value = params.get(key)
    require(isinstance(value, bool), f"hermes_lcm {key} must be a boolean")
    return value


def _bounded_int(
    params: dict[str, object],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = params.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"hermes_lcm {key} must be an integer",
    )
    require(minimum <= value <= maximum, f"hermes_lcm {key} must be between {minimum} and {maximum}")
    return value


def _bounded_float(
    params: dict[str, object],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = params.get(key)
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"hermes_lcm {key} must be a number",
    )
    number = float(value)
    require(minimum <= number <= maximum, f"hermes_lcm {key} must be between {minimum} and {maximum}")
    return number


@register_memory
class HermesLCMMemory(Memory):
    """Thin official adapter over the product-owned ``TrajectoryStore``."""

    memory_type = "hermes_lcm"
    supports_parallel_query = False

    _ALLOWED_PARAMS = frozenset({
        "workspace_root",
        "trajectories_root_dir",
        "dataset_name",
        "dataset_revision",
        "harness_commit",
        "tier",
        "domain",
        "candidate_limit",
        "max_text_items",
        "max_text_chars_per_item",
        "max_image_items",
        "include_adjacent",
        "protect_sensitive",
        "semantic_enabled",
        "semantic_provider",
        "semantic_model",
        "semantic_top_trajectories",
        "semantic_build_timeout_seconds",
        "semantic_query_timeout_seconds",
    })

    def __init__(self, memory_params: dict[str, object]) -> None:
        unexpected = sorted(set(memory_params) - self._ALLOWED_PARAMS)
        require(not unexpected, f"hermes_lcm memory_params contains unexpected keys: {unexpected}")
        semantic_enabled = bool(memory_params.get("semantic_enabled", False))
        require(
            isinstance(memory_params.get("semantic_enabled", False), bool),
            "hermes_lcm semantic_enabled must be a boolean",
        )
        semantic_provider = str(memory_params.get("semantic_provider", "") or "").strip()
        semantic_model = str(memory_params.get("semantic_model", "") or "").strip()
        if semantic_enabled:
            require(semantic_provider, "hermes_lcm semantic_provider must be set when enabled")
            require(semantic_model, "hermes_lcm semantic_model must be set when enabled")
        semantic_params = dict(memory_params)
        semantic_params.setdefault("semantic_top_trajectories", 12)
        semantic_params.setdefault("semantic_build_timeout_seconds", 120.0)
        semantic_params.setdefault("semantic_query_timeout_seconds", 5.0)
        normalized: dict[str, object] = {
            "workspace_root": str(Path(_required_text(memory_params, "workspace_root")).expanduser().resolve()),
            "trajectories_root_dir": str(Path(_required_text(memory_params, "trajectories_root_dir")).expanduser().resolve()),
            "dataset_name": _required_text(memory_params, "dataset_name"),
            "dataset_revision": _required_text(memory_params, "dataset_revision"),
            "harness_commit": _required_text(memory_params, "harness_commit"),
            "tier": _required_text(memory_params, "tier"),
            "domain": _required_text(memory_params, "domain"),
            "candidate_limit": _bounded_int(memory_params, "candidate_limit", minimum=1, maximum=128),
            "max_text_items": _bounded_int(memory_params, "max_text_items", minimum=1, maximum=16),
            "max_text_chars_per_item": _bounded_int(
                memory_params,
                "max_text_chars_per_item",
                minimum=256,
                maximum=8000,
            ),
            "max_image_items": _bounded_int(memory_params, "max_image_items", minimum=0, maximum=8),
            "include_adjacent": _required_bool(memory_params, "include_adjacent"),
            "protect_sensitive": _required_bool(memory_params, "protect_sensitive"),
            "semantic_enabled": semantic_enabled,
            "semantic_provider": semantic_provider,
            "semantic_model": semantic_model,
            "semantic_top_trajectories": _bounded_int(
                semantic_params,
                "semantic_top_trajectories",
                minimum=1,
                maximum=32,
            ),
            "semantic_build_timeout_seconds": _bounded_float(
                semantic_params,
                "semantic_build_timeout_seconds",
                minimum=0.1,
                maximum=600.0,
            ),
            "semantic_query_timeout_seconds": _bounded_float(
                semantic_params,
                "semantic_query_timeout_seconds",
                minimum=0.1,
                maximum=30.0,
            ),
        }
        workspace_root = Path(str(normalized["workspace_root"]))
        data_root = Path(str(normalized["trajectories_root_dir"]))
        require(data_root.is_dir(), "hermes_lcm trajectories_root_dir must exist")
        workspace_root.mkdir(parents=True, exist_ok=True)
        super().__init__(normalized)
        self._api = _product_api()
        self._store = None
        self._db_path: Path | None = None
        self._ordered_ids: list[str] = []
        self._last_query_metadata: dict[str, object] = {}
        self._semantic_ready = False
        self._semantic_setup_status = (
            "pending" if bool(normalized["semantic_enabled"]) else "disabled"
        )
        self._semantic_setup_error: str | None = None
        self._semantic_setup_fallbacks = 0
        self.last_query_latency_seconds = 0.0
        self.read_only = False
        # Runtime-only side-channel trace root (never persisted into memory
        # config, so save/load config equality -- and thus run isolation --
        # is unchanged). Delivered by the harness via configure_runtime.
        self._query_trace_dir: Path | None = None
        # Count of best-effort telemetry writes that failed (disk full,
        # unwritable path, ...). A telemetry write must NEVER fail a question.
        self._telemetry_write_failures = 0
        # H3.1 composition-seam overrides (env-var only, see the
        # _arm_quota_from_env / _lexical_floor_from_env docstrings); None / 0
        # reproduce current bytes. Together they select the A+D hybrid.
        self._arm_quota: tuple[int, int] | None = _arm_quota_from_env()
        self._lexical_floor: int = _lexical_floor_from_env()

    @classmethod
    def reconcile_loaded_memory_config(
        cls,
        saved_config: dict[str, object],
        requested_config: dict[str, object] | None,
    ) -> dict[str, object]:
        """Allow only non-semantic local paths to rebase on artifact restore."""
        require(saved_config.get("memory_type") == cls.memory_type, "saved memory type mismatch")
        if requested_config is None:
            return {
                "memory_type": cls.memory_type,
                "memory_params": dict(saved_config["memory_params"]),
            }
        require(requested_config.get("memory_type") == cls.memory_type, "requested memory type mismatch")
        saved_params = dict(saved_config["memory_params"])
        requested_params = dict(requested_config["memory_params"])
        semantic_defaults: dict[str, object] = {
            "semantic_enabled": False,
            "semantic_provider": "",
            "semantic_model": "",
            "semantic_top_trajectories": 12,
            "semantic_build_timeout_seconds": 120.0,
            "semantic_query_timeout_seconds": 5.0,
        }
        for key, value in semantic_defaults.items():
            saved_params.setdefault(key, value)
            requested_params.setdefault(key, value)
        runtime_paths = {"workspace_root", "trajectories_root_dir"}
        require(
            {key: value for key, value in saved_params.items() if key not in runtime_paths}
            == {key: value for key, value in requested_params.items() if key not in runtime_paths},
            "hermes_lcm immutable corpus and retrieval parameters must match on load",
        )
        require(
            set(saved_params) == set(requested_params),
            "hermes_lcm requested config keys must match the saved config",
        )
        return {
            "memory_type": cls.memory_type,
            "memory_params": requested_params,
        }

    def _identity(self):
        return self._api.CorpusIdentity(
            dataset_name=str(self.memory_params["dataset_name"]),
            dataset_revision=str(self.memory_params["dataset_revision"]),
            harness_commit=str(self.memory_params["harness_commit"]),
            tier=str(self.memory_params["tier"]),
            domain=str(self.memory_params["domain"]),
            ingest_config_digest="trajectory-adapter-v1",
        )

    def _ensure_writable_store(self):
        if self._store is not None:
            require(not self.read_only, "loaded Hermes-LCM memory is read-only")
            return self._store
        workspace_root = Path(str(self.memory_params["workspace_root"]))
        instance_dir = Path(tempfile.mkdtemp(prefix="corpus-build-", dir=workspace_root))
        self._db_path = instance_dir / "lcm.db"
        self._store = self._api.TrajectoryStore(
            self._db_path,
            self._identity(),
            asset_root=Path(str(self.memory_params["trajectories_root_dir"])),
            protect_sensitive=bool(self.memory_params["protect_sensitive"]),
            semantic_top_trajectories=int(
                self.memory_params["semantic_top_trajectories"]
            ),
        )
        return self._store

    def insert(self, trajectory: dict[str, object]) -> None:
        store = self._ensure_writable_store()
        prepared = prepare_trajectory_insert(
            trajectory,
            trajectories_root_dir=Path(str(self.memory_params["trajectories_root_dir"])),
        )
        states: list[Any] = []
        for state, screenshot_path in zip(
            prepared.simplified["states"], prepared.screenshot_sources
        ):
            require(isinstance(state, dict), "prepared trajectory state must be an object")
            states.append(self._api.TrajectoryState(
                state_index=int(state["state_index"]),
                step=int(state["step"]),
                url=str(state["url"]),
                incoming_action=(
                    str(state["action"]) if state.get("action") is not None else None
                ),
                thoughts=(
                    str(state["thoughts"]) if state.get("thoughts") is not None else None
                ),
                text=str(state["text"]),
                screenshot_path=screenshot_path,
                observed_at=None,
                observed_at_source=None,
                occurred_at=None,
                occurred_at_source=None,
            ))
        source = self._api.TrajectorySource(
            trajectory_id=prepared.trajectory_id,
            ordinal=len(self._ordered_ids),
            goal=str(prepared.simplified["goal"]),
            start_url=str(prepared.simplified["start_url"]),
            outcome=(
                str(prepared.simplified["outcome"])
                if prepared.simplified.get("outcome") is not None
                else None
            ),
            states=tuple(states),
            source_payload=prepared.simplified,
        )
        result = store.insert(source)
        if not result.already_current:
            self._ordered_ids.append(prepared.trajectory_id)
        elif prepared.trajectory_id not in self._ordered_ids:
            self._ordered_ids.append(prepared.trajectory_id)

    def _finalize(self) -> None:
        require(self._store is not None, "Hermes-LCM memory has no inserted trajectories")
        if self._store.status != "complete":
            require(not self.read_only, "loaded Hermes-LCM corpus is incomplete")
            self._store.finalize(self._ordered_ids)
        if bool(self.memory_params["semantic_enabled"]) and not self._semantic_ready:
            try:
                if self.read_only:
                    provider = self._api.create_trajectory_embedding_provider(
                        str(self.memory_params["semantic_provider"]),
                        str(self.memory_params["semantic_model"]),
                        timeout_seconds=float(
                            self.memory_params["semantic_query_timeout_seconds"]
                        ),
                        for_backfill=False,
                    )
                    self._store.set_embedding_provider(provider)
                else:
                    build_provider = self._api.create_trajectory_embedding_provider(
                        str(self.memory_params["semantic_provider"]),
                        str(self.memory_params["semantic_model"]),
                        timeout_seconds=float(
                            self.memory_params["semantic_build_timeout_seconds"]
                        ),
                        for_backfill=True,
                    )
                    self._store.build_semantic_index(build_provider)
                    query_provider = self._api.create_trajectory_embedding_provider(
                        str(self.memory_params["semantic_provider"]),
                        str(self.memory_params["semantic_model"]),
                        timeout_seconds=float(
                            self.memory_params["semantic_query_timeout_seconds"]
                        ),
                        for_backfill=False,
                    )
                    self._store.set_embedding_provider(query_provider)
                self._semantic_setup_status = "ready"
                self._semantic_setup_error = None
            except Exception as exc:
                self._store.set_embedding_provider(None)
                self._semantic_setup_status = "fallback"
                self._semantic_setup_error = type(exc).__name__
                self._semantic_setup_fallbacks = 1
            self._semantic_ready = True

    def configure_runtime(self, **kwargs: object) -> None:
        """Apply non-persisted runtime overrides (harness-supplied trace root).

        ``query_trace_dir`` is the run-root side-channel directory where the
        harness collects per-question artifacts. It is runtime-only and never
        enters saved memory config, so it cannot perturb run isolation or the
        reader prompt bytes.
        """
        query_trace_dir = kwargs.get("query_trace_dir")
        if query_trace_dir is not None:
            if isinstance(query_trace_dir, Path):
                self._query_trace_dir = query_trace_dir.resolve()
            else:
                require(
                    isinstance(query_trace_dir, str) and query_trace_dir.strip(),
                    "hermes_lcm query_trace_dir override must be a non-empty string or Path",
                )
                self._query_trace_dir = Path(query_trace_dir).resolve()

    def _guard_config_echo(self) -> dict[str, object] | None:
        """Echo the resolved query-path spend-guard for run summaries (#123)."""
        provider = getattr(self._store, "embedding_provider", None) if self._store else None
        guard = getattr(provider, "spend_guard", None)
        if guard is None:
            return None
        return {
            "max_calls": getattr(guard, "max_calls", None),
            "window_seconds": getattr(guard, "window_seconds", None),
            "backoff_seconds": getattr(guard, "backoff_seconds", None),
        }

    def _write_query_trace(self, exact_refs: list[str]) -> None:
        """Persist the per-query semantic telemetry as a run-root side-channel
        file (#124 e). Written strictly AFTER the product ``query()`` returns
        and never touches the returned evidence or the post-query metadata, so
        reader prompt bytes stay byte-identical to baseline. Product methods are
        read defensively so an older product simply yields a null-populated
        record instead of raising.

        The ENTIRE body is fenced: this runs inside ``query()`` before the
        question returns, so any I/O failure (disk full, unwritable dir, a file
        where a directory is expected) must degrade to a counted, logged warning
        -- never fail the question."""
        if self._query_trace_dir is None or self._store is None:
            return
        try:
            context = self.get_query_context()
            question_id = context.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                return

            def _call(name: str):
                fn = getattr(self._store, name, None)
                return fn() if callable(fn) else None

            telemetry = _call("last_query_telemetry") or {}
            record = {
                "backend": self.memory_type,
                "corpus_uid": self._store.corpus_uid,
                "question_id": question_id,
                "guard_config": self._guard_config_echo(),
                "arm_quota": list(self._arm_quota) if self._arm_quota is not None else None,
                "lexical_floor": self._lexical_floor,
                "semantic_attempt": _call("last_semantic_attempt"),
                "semantic_attempt_counters": _call("semantic_attempt_counters"),
                "source_candidate_ranks": telemetry.get("source_candidate_ranks", []),
                "state_candidate_pool": telemetry.get("state_candidate_pool", []),
                "delivered_evidence_refs": telemetry.get(
                    "delivered_evidence_refs", list(exact_refs)
                ),
            }
            out_dir = self._query_trace_dir / question_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "hermes_lcm_semantic_telemetry.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never fail a query
            self._telemetry_write_failures += 1
            print(
                f"[hermes_lcm] telemetry write skipped ({type(exc).__name__}): {exc}",
                file=sys.stderr,
                flush=True,
            )

    def run_summary(self) -> dict[str, object]:
        """Aggregate semantic-funnel counters + guard echo for the run (#124 f)."""
        counters: dict[str, object] | None = None
        if self._store is not None:
            fn = getattr(self._store, "semantic_attempt_counters", None)
            if callable(fn):
                counters = fn()
        return {
            "backend": self.memory_type,
            "domain": str(self.memory_params.get("domain", "")),
            "semantic_attempt_counters": counters,
            "guard_config": self._guard_config_echo(),
            "arm_quota": list(self._arm_quota) if self._arm_quota is not None else None,
            "lexical_floor": self._lexical_floor,
            "telemetry_write_failures": self._telemetry_write_failures,
        }

    def emit_run_summary(self) -> dict[str, object]:
        """Print one loud, unmissable run-summary line and return the payload."""
        summary = self.run_summary()
        counters = summary.get("semantic_attempt_counters") or {}
        guard = summary.get("guard_config") or {}
        print(
            "[hermes_lcm run-summary] "
            f"domain={summary['domain']} "
            f"semantic_attempts={counters.get('successes', 0)}/{counters.get('attempts', 0)} "
            f"fallbacks={counters.get('fallbacks', 0)} "
            f"by_reason={counters.get('fallbacks_by_reason', {})} "
            f"telemetry_write_failures={summary['telemetry_write_failures']} "
            f"guard(max_calls={guard.get('max_calls')},"
            f"window_s={guard.get('window_seconds')},"
            f"backoff_s={guard.get('backoff_seconds')})",
            flush=True,
        )
        return summary

    @staticmethod
    def _render_hit(hit) -> str:
        lines = [
            f"[{hit.exact_ref}]",
            f"Trajectory: {hit.trajectory_id}",
            f"Goal: {hit.goal}",
            f"Outcome: {hit.outcome or '<unknown>'}",
            f"State: {hit.state_index} (sequence {hit.sequence_ordinal}, step {hit.step})",
            f"URL: {hit.url}",
            f"Incoming action: {hit.incoming_action or '<none>'}",
        ]
        if hit.thoughts:
            lines.append(f"Thought: {hit.thoughts}")
        excerpt_label = "Visible state"
        if hit.text_truncated:
            excerpt_label = f"Visible state excerpt (offset {hit.text_offset})"
        lines.append(f"{excerpt_label}: {hit.text}")
        if hit.observed_at is not None:
            lines.append(
                f"Observed at: {hit.observed_at} (source: {hit.observed_at_source})"
            )
        if hit.occurred_at is not None:
            lines.append(
                f"Occurred at: {hit.occurred_at} (source: {hit.occurred_at_source})"
            )
        return "\n".join(lines)

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        del query_image  # input-only; never persisted, traced, or returned
        self._finalize()
        started = time.perf_counter()
        query_kwargs: dict[str, object] = {
            "candidate_limit": int(self.memory_params["candidate_limit"]),
            "limit": int(self.memory_params["max_text_items"]),
            "image_limit": int(self.memory_params["max_image_items"]),
            "include_adjacent": bool(self.memory_params["include_adjacent"]),
            "text_char_limit": int(self.memory_params["max_text_chars_per_item"]),
        }
        if self._arm_quota is not None:
            # Only products that implement Option D (bench/h3.1-composition
            # and later) accept this kwarg; omitted entirely when unset so a
            # pre-composition product checkout's query() never sees it.
            query_kwargs["arm_quota"] = self._arm_quota
        if self._lexical_floor > 0:
            # A+D hybrid floor; same omitted-when-unset discipline as arm_quota
            # so a pre-composition product checkout's query() never sees it.
            query_kwargs["lexical_floor"] = self._lexical_floor
        hits = self._store.query(query, **query_kwargs)
        context: list[MemoryContextItem] = []
        for hit in hits:
            context.append({"type": "text", "value": self._render_hit(hit)})
            if hit.screenshot_path is not None:
                context.append({"type": "image", "value": hit.screenshot_path})
        self.last_query_latency_seconds = time.perf_counter() - started
        semantic_metrics = self._store.semantic_metrics()
        self._last_query_metadata = {
            "backend": self.memory_type,
            "corpus_uid": self._store.corpus_uid,
            "exact_refs": [hit.exact_ref for hit in hits],
            "query_digest": self._store.query_digest(hits),
            "text_items": len(hits),
            "image_items": sum(hit.screenshot_path is not None for hit in hits),
            "semantic_enabled": bool(self.memory_params["semantic_enabled"]),
            "semantic_provider": str(self.memory_params["semantic_provider"]),
            "semantic_model": str(self.memory_params["semantic_model"]),
            "semantic_setup_status": self._semantic_setup_status,
            "semantic_setup_error": self._semantic_setup_error,
            "embedding_document_calls": semantic_metrics["document_calls"],
            "embedding_document_tokens": semantic_metrics["document_tokens"],
            "embedding_query_calls": semantic_metrics["query_calls"],
            "embedding_query_tokens": semantic_metrics["query_tokens"],
            "embedding_fallbacks": (
                semantic_metrics["fallbacks"] + self._semantic_setup_fallbacks
            ),
            "provider_calls": (
                semantic_metrics["document_calls"]
                + semantic_metrics["query_calls"]
            ),
        }
        # Side-channel telemetry only: does not mutate `context` (reader prompt
        # bytes) or `_last_query_metadata` (post-query metadata stays
        # deterministic across identical queries).
        self._write_query_trace([hit.exact_ref for hit in hits])
        return context

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
        self._finalize()
        require(not self.read_only, "loaded read-only Hermes-LCM memory cannot be re-saved")
        database_path = output_dir / "lcm.db"
        self._store.backup_to(database_path)
        (output_dir / "corpus-manifest.json").write_text(
            json.dumps(self._store.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _load_backend(self, input_dir: Path) -> None:
        database_path = input_dir / "lcm.db"
        manifest_path = input_dir / "corpus-manifest.json"
        require(database_path.is_file(), "saved Hermes-LCM memory is missing lcm.db")
        require(manifest_path.is_file(), "saved Hermes-LCM memory is missing corpus-manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(isinstance(manifest, dict), "saved Hermes-LCM corpus manifest must be an object")
        manifest.setdefault("semantic_index", None)
        if self._store is not None:
            self._store.close()
        self._db_path = database_path
        self._store = self._api.TrajectoryStore(
            database_path,
            self._identity(),
            asset_root=Path(str(self.memory_params["trajectories_root_dir"])),
            read_only=True,
            protect_sensitive=bool(self.memory_params["protect_sensitive"]),
            semantic_top_trajectories=int(
                self.memory_params["semantic_top_trajectories"]
            ),
        )
        require(self._store.manifest() == manifest, "saved Hermes-LCM corpus manifest mismatch")
        self.read_only = True
        self._semantic_ready = False

    def close(self) -> None:
        if self._store is not None:
            # Emit the loud run summary only for a process that actually served
            # queries (the reader stage), so the build process stays quiet.
            counters = self.run_summary().get("semantic_attempt_counters") or {}
            if counters.get("attempts", 0):
                self.emit_run_summary()
            self._store.close()
            self._store = None
