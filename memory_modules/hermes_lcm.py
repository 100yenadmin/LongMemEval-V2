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
    })

    def __init__(self, memory_params: dict[str, object]) -> None:
        unexpected = sorted(set(memory_params) - self._ALLOWED_PARAMS)
        require(not unexpected, f"hermes_lcm memory_params contains unexpected keys: {unexpected}")
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
        self.last_query_latency_seconds = 0.0
        self.read_only = False

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
        hits = self._store.query(
            query,
            candidate_limit=int(self.memory_params["candidate_limit"]),
            limit=int(self.memory_params["max_text_items"]),
            image_limit=int(self.memory_params["max_image_items"]),
            include_adjacent=bool(self.memory_params["include_adjacent"]),
            text_char_limit=int(self.memory_params["max_text_chars_per_item"]),
        )
        context: list[MemoryContextItem] = []
        for hit in hits:
            context.append({"type": "text", "value": self._render_hit(hit)})
            if hit.screenshot_path is not None:
                context.append({"type": "image", "value": hit.screenshot_path})
        self.last_query_latency_seconds = time.perf_counter() - started
        self._last_query_metadata = {
            "backend": self.memory_type,
            "corpus_uid": self._store.corpus_uid,
            "exact_refs": [hit.exact_ref for hit in hits],
            "query_digest": self._store.query_digest(hits),
            "text_items": len(hits),
            "image_items": sum(hit.screenshot_path is not None for hit in hits),
            "provider_calls": 0,
        }
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
        if self._store is not None:
            self._store.close()
        self._db_path = database_path
        self._store = self._api.TrajectoryStore(
            database_path,
            self._identity(),
            asset_root=Path(str(self.memory_params["trajectories_root_dir"])),
            read_only=True,
            protect_sensitive=bool(self.memory_params["protect_sensitive"]),
        )
        require(self._store.manifest() == manifest, "saved Hermes-LCM corpus manifest mismatch")
        self.read_only = True

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
