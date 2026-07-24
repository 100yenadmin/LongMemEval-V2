#!/usr/bin/env python3
"""Read-only, haystack-scoped search CLI for a Hermes-LCM trajectory store."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys
from typing import Any, Iterable, Sequence
from urllib.parse import quote


STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "did",
        "do",
        "does",
        "for",
        "from",
        "happen",
        "happened",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "the",
        "then",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
    }
)
QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
SPLIT_PUNCT_RE = re.compile(r"[-:/]+")
STRIP_EDGE_PUNCT = "\"'()[]{}.,;"
BOOLEAN_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
MAX_CANDIDATES = 128
MAX_RESULTS = 24
MAX_TEXT_CHARS = 8_000

PROVIDER_ENV_REQUIREMENTS: dict[str, str] = {
    # Keep in sync with hermes_lcm_agentic.py's SEMANTIC_PROVIDER_ENV_VARS.
    # This file is copied standalone into the codex sandbox (it cannot
    # import the rest of the package there), so it carries its own copy of
    # the provider -> required-credential mapping.
    "voyage": "VOYAGE_API_KEY",
}


class SemanticConfigError(RuntimeError):
    """Semantic retrieval is enabled but its required credential is absent
    from the sandbox's environment. This is a config error, not a
    transient provider blip: callers must fail loud, never silently
    degrade to FTS-only retrieval for a missing key
    (100yenadmin/hermes-lcm#147 P3 revision)."""


def semantic_self_check(scope: dict[str, Any]) -> dict[str, Any]:
    """Report semantic-retrieval availability without making any network
    call. A `missing_key` status is the config-error class that must fail
    loud (see SemanticConfigError); it is distinct from a
    `fallback:<ExceptionType>` status discovered later at call time inside
    semantic_source_ranks, which stays a counted-but-tolerated transient
    provider failure."""
    semantic_enabled = bool(scope.get("semantic_enabled", False))
    provider_name = str(scope.get("semantic_provider", "")).strip() or None
    required_env_var = (
        PROVIDER_ENV_REQUIREMENTS.get(provider_name) if provider_name else None
    )
    env_var_present = bool(
        required_env_var and os.environ.get(required_env_var, "").strip()
    )
    if not semantic_enabled:
        status = "disabled"
    elif required_env_var is None:
        status = "unknown_provider_requirement"
    elif env_var_present:
        status = "available"
    else:
        status = "missing_key"
    return {
        "semantic_enabled": semantic_enabled,
        "provider": provider_name,
        "required_env_var": required_env_var,
        "env_var_present": env_var_present,
        "status": status,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_scope(scope_path: Path) -> dict[str, Any]:
    payload = load_json(scope_path)
    require(isinstance(payload, dict), "hermes_scope.json must contain an object")
    trajectory_ids = payload.get("trajectory_ids")
    require(
        isinstance(trajectory_ids, list)
        and trajectory_ids
        and all(isinstance(item, str) and item for item in trajectory_ids),
        "hermes_scope.json trajectory_ids must be a non-empty string list",
    )
    require(
        len(set(trajectory_ids)) == len(trajectory_ids),
        "hermes_scope.json trajectory_ids must not contain duplicates",
    )
    for key in ("canonical_store_path", "asset_root", "product_root"):
        value = payload.get(key)
        require(
            isinstance(value, str) and value.strip(),
            f"hermes_scope.json {key} must be a non-empty string",
        )
    return dict(payload)


def connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    require(path.is_file(), f"Canonical Hermes-LCM store does not exist: {path}")
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=5.0,
        check_same_thread=False,
        isolation_level=None,
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.row_factory = sqlite3.Row
    return connection


def resolve_scope_sources(
    connection: sqlite3.Connection,
    trajectory_ids: Sequence[str],
) -> dict[str, sqlite3.Row]:
    placeholders = ",".join("?" for _ in trajectory_ids)
    rows = connection.execute(
        f"""
        SELECT source_id, trajectory_id, ordinal, goal, start_url, outcome, state_count
        FROM lcm_trajectory_sources
        WHERE trajectory_id IN ({placeholders})
        """,
        tuple(trajectory_ids),
    ).fetchall()
    by_id = {str(row["trajectory_id"]): row for row in rows}
    missing = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id not in by_id]
    require(
        not missing,
        "Canonical Hermes-LCM store is missing scoped trajectory ids: "
        + ", ".join(missing),
    )
    return by_id


def extract_search_terms(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    terms: list[str] = []
    for phrase in QUOTED_PHRASE_RE.findall(text):
        cleaned = phrase.strip()
        if cleaned:
            terms.append(cleaned)
    text_without_phrases = QUOTED_PHRASE_RE.sub(" ", text)
    for token in text_without_phrases.split():
        cleaned = token.strip().strip(STRIP_EDGE_PUNCT)
        if not cleaned or cleaned.upper() in BOOLEAN_OPERATORS:
            continue
        terms.append(cleaned)
        if SPLIT_PUNCT_RE.search(cleaned):
            parts = [part for part in SPLIT_PUNCT_RE.split(cleaned) if part]
            if len(parts) > 1:
                terms.extend(parts)
    if not terms:
        fallback = text.strip().strip(STRIP_EDGE_PUNCT)
        if fallback:
            terms.append(fallback)
    return list(dict.fromkeys(terms))


def fts_expression(query: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in extract_search_terms(query):
        normalized = raw.casefold().strip()
        if len(normalized) < 2 or normalized in STOPWORDS or normalized in seen:
            continue
        safe = normalized.replace('"', '""')
        if not any(character.isalnum() for character in safe):
            continue
        seen.add(normalized)
        terms.append(f'"{safe}"')
        if len(terms) >= 16:
            break
    return " OR ".join(terms)


def fts_rows(
    connection: sqlite3.Connection,
    expression: str,
    candidate_limit: int,
    *,
    source_ids: Sequence[int],
) -> list[sqlite3.Row]:
    require(source_ids, "FTS query requires a non-empty source scope")
    placeholders = ",".join("?" for _ in source_ids)
    params: list[Any] = [expression]
    params.extend(int(source_id) for source_id in source_ids)
    params.append(int(candidate_limit))
    return connection.execute(
        f"""
        SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
               a.relative_path, a.sha256 AS asset_sha256,
               bm25(lcm_trajectory_states_fts) AS rank
        FROM lcm_trajectory_states_fts
        JOIN lcm_trajectory_states s
          ON s.state_id = lcm_trajectory_states_fts.rowid
        JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
        LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
        WHERE lcm_trajectory_states_fts MATCH ?
          AND s.source_id IN ({placeholders})
        ORDER BY rank ASC, src.ordinal ASC, s.sequence_ordinal ASC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()


def load_product_api(product_root: str | Path):
    root = Path(product_root).expanduser().resolve()
    init_path = root / "__init__.py"
    require(init_path.is_file(), f"Product root is not a Hermes-LCM checkout: {root}")
    alias = "_hermes_lcm_search_runtime"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias,
            str(init_path),
            submodule_search_locations=[str(root)],
        )
        require(
            spec is not None and spec.loader is not None,
            "Could not construct Hermes-LCM package spec",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{alias}.trajectory_store")


def normalized_vector(
    values: Sequence[float],
    *,
    expected_dim: int | None = None,
) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    require(vector and all(math.isfinite(value) for value in vector), "Invalid query vector")
    if expected_dim is not None:
        require(len(vector) == expected_dim, "Query embedding dimension changed")
    norm = math.sqrt(sum(value * value for value in vector))
    require(math.isfinite(norm) and norm > 0.0, "Query vector has invalid norm")
    return tuple(value / norm for value in vector)


def unpack_vector(payload: bytes, dim: int) -> tuple[float, ...]:
    require(len(payload) == dim * 4, "Stored trajectory embedding dimension is invalid")
    return tuple(struct.unpack(f"<{dim}f", payload))


def semantic_source_ranks(
    connection: sqlite3.Connection,
    query: str,
    *,
    source_ids: Sequence[int],
    scope: dict[str, Any],
) -> tuple[list[tuple[int, float]], str]:
    self_check = semantic_self_check(scope)
    if not self_check["semantic_enabled"]:
        return [], "disabled"
    if self_check["status"] == "missing_key":
        raise SemanticConfigError(
            "semantic_enabled=True (provider="
            f"{self_check['provider']!r}) but required credential "
            f"{self_check['required_env_var']} is not set in this "
            "sandbox's environment -- config error, not a transient "
            "provider blip; refusing to silently degrade to FTS-only "
            "retrieval."
        )
    profile = connection.execute(
        """
        SELECT * FROM lcm_trajectory_embedding_profiles
        WHERE active = 1
        """
    ).fetchone()
    if profile is None:
        return [], "missing_profile"
    provider_name = str(scope.get("semantic_provider", "")).strip()
    model_name = str(scope.get("semantic_model", "")).strip()
    require(provider_name and model_name, "Semantic provider/model must be configured")
    if (
        str(profile["provider"]) != provider_name
        or str(profile["model_name"]) != model_name
    ):
        return [], "profile_mismatch"
    try:
        api = load_product_api(str(scope["product_root"]))
        provider = api.create_trajectory_embedding_provider(
            provider_name,
            model_name,
            timeout_seconds=float(scope.get("semantic_query_timeout_seconds", 5.0)),
            for_backfill=False,
        )
        query_vector = normalized_vector(
            provider.embed_query(query),
            expected_dim=int(profile["dim"]),
        )
        placeholders = ",".join("?" for _ in source_ids)
        rows = connection.execute(
            f"""
            SELECT source_id, vector
            FROM lcm_trajectory_embeddings
            WHERE profile_digest = ?
              AND source_id IN ({placeholders})
            """,
            (str(profile["profile_digest"]), *(int(item) for item in source_ids)),
        ).fetchall()
        ranked: list[tuple[int, float]] = []
        for row in rows:
            vector = unpack_vector(bytes(row["vector"]), int(profile["dim"]))
            similarity = sum(left * right for left, right in zip(query_vector, vector))
            ranked.append((int(row["source_id"]), float(similarity)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        top_n = min(max(1, int(scope.get("semantic_top_trajectories", 12))), 32)
        return ranked[:top_n], "success"
    except Exception as exc:  # product query degrades semantic failures to FTS
        return [], f"fallback:{type(exc).__name__}"


def select_diverse(rows: Iterable[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
    selected: list[sqlite3.Row] = []
    per_trajectory: dict[str, int] = {}
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        if per_trajectory.get(trajectory_id, 0) >= 5:
            continue
        selected.append(row)
        per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def exact_excerpt(text: str, query: str, limit: int) -> tuple[str, int, bool]:
    if len(text) <= limit:
        return text, 0, False
    folded = text.casefold()
    positions = [
        folded.find(term.casefold())
        for term in extract_search_terms(query)
        if len(term.strip()) >= 2 and folded.find(term.casefold()) >= 0
    ]
    first_match = min(positions) if positions else 0
    start = max(0, first_match - (limit // 3))
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    return text[start:end], start, True


def corpus_uid(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT corpus_uid, status FROM lcm_trajectory_corpora WHERE singleton = 1"
    ).fetchone()
    require(row is not None and row["status"] == "complete", "Canonical corpus is not complete")
    value = row["corpus_uid"]
    require(isinstance(value, str) and value, "Canonical corpus has no corpus_uid")
    return value


def search_store(
    connection: sqlite3.Connection,
    query: str,
    *,
    scope: dict[str, Any],
    limit: int,
    candidate_limit: int,
) -> dict[str, Any]:
    sources = resolve_scope_sources(connection, scope["trajectory_ids"])
    source_ids = [int(sources[item]["source_id"]) for item in scope["trajectory_ids"]]
    expression = fts_expression(query)
    if not expression:
        return {
            "query": query,
            "scope_trajectory_count": len(source_ids),
            "semantic_status": "not_attempted",
            "hits": [],
        }
    global_rows = fts_rows(
        connection,
        expression,
        candidate_limit,
        source_ids=source_ids,
    )
    semantic_ranks, semantic_status = semantic_source_ranks(
        connection,
        query,
        source_ids=source_ids,
        scope=scope,
    )
    semantic_source_ids = [source_id for source_id, _score in semantic_ranks]
    scoped_rows = (
        fts_rows(
            connection,
            expression,
            candidate_limit,
            source_ids=semantic_source_ids,
        )
        if semantic_source_ids
        else []
    )

    if semantic_ranks and scoped_rows:
        row_by_id: dict[int, sqlite3.Row] = {}
        score_by_candidate: dict[int, float] = {}
        semantic_position = {
            source_id: position
            for position, (source_id, _score) in enumerate(semantic_ranks, start=1)
        }
        for position, row in enumerate(global_rows, start=1):
            state_id = int(row["state_id"])
            row_by_id[state_id] = row
            score_by_candidate[state_id] = score_by_candidate.get(state_id, 0.0) + (
                1.0 / (60.0 + position)
            )
        for position, row in enumerate(scoped_rows, start=1):
            state_id = int(row["state_id"])
            row_by_id[state_id] = row
            score_by_candidate[state_id] = score_by_candidate.get(state_id, 0.0) + (
                1.0 / (60.0 + position)
            )
            trajectory_position = semantic_position.get(int(row["source_id"]), 32)
            score_by_candidate[state_id] += 1.0 / (60.0 + trajectory_position)
        rows = sorted(
            row_by_id.values(),
            key=lambda row: (
                -score_by_candidate[int(row["state_id"])],
                int(row["ordinal"]),
                int(row["sequence_ordinal"]),
            ),
        )
        candidate_kind = {
            int(row["state_id"]): "semantic_fts" for row in scoped_rows
        }
        candidate_score = {
            state_id: -score for state_id, score in score_by_candidate.items()
        }
    else:
        rows = global_rows
        candidate_kind = {}
        candidate_score = {
            int(row["state_id"]): float(row["rank"]) for row in rows
        }

    include_adjacent = bool(scope.get("include_adjacent", True))
    adjacent_reserve = min(6, limit // 3) if include_adjacent else 0
    nucleus_limit = max(1, limit - adjacent_reserve)
    selected = select_diverse(rows, nucleus_limit)
    selected_ids = {int(row["state_id"]) for row in selected}
    match_kind_by_id = {
        int(row["state_id"]): candidate_kind.get(int(row["state_id"]), "fts")
        for row in selected
    }
    score_by_id = {
        int(row["state_id"]): candidate_score[int(row["state_id"])]
        for row in selected
    }
    if include_adjacent and selected and len(selected) < limit:
        nucleus_rows = list(selected)
        adjacent_by_nucleus: list[list[sqlite3.Row]] = []
        for nucleus in nucleus_rows:
            adjacent_rows = connection.execute(
                """
                SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                       a.relative_path, a.sha256 AS asset_sha256, 0.0 AS rank
                FROM lcm_trajectory_states s
                JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
                LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
                WHERE s.source_id = ? AND s.sequence_ordinal IN (?, ?)
                ORDER BY ABS(s.sequence_ordinal - ?), s.sequence_ordinal
                """,
                (
                    int(nucleus["source_id"]),
                    int(nucleus["sequence_ordinal"]) - 1,
                    int(nucleus["sequence_ordinal"]) + 1,
                    int(nucleus["sequence_ordinal"]),
                ),
            ).fetchall()
            adjacent_by_nucleus.append(list(adjacent_rows))
        while len(selected) < limit and any(adjacent_by_nucleus):
            made_progress = False
            for nucleus, adjacent_rows in zip(nucleus_rows, adjacent_by_nucleus):
                while adjacent_rows:
                    row = adjacent_rows.pop(0)
                    state_id = int(row["state_id"])
                    if state_id in selected_ids:
                        continue
                    selected.append(row)
                    selected_ids.add(state_id)
                    match_kind_by_id[state_id] = "adjacent"
                    score_by_id[state_id] = score_by_id[int(nucleus["state_id"])] + 0.000001
                    made_progress = True
                    break
                if len(selected) >= limit:
                    break
            if not made_progress:
                break

    uid = corpus_uid(connection)
    text_char_limit = min(
        max(256, int(scope.get("text_char_limit", 2_000))),
        MAX_TEXT_CHARS,
    )
    hits: list[dict[str, Any]] = []
    for rank, row in enumerate(selected[:limit], start=1):
        text, text_offset, text_truncated = exact_excerpt(
            str(row["text"]),
            query,
            text_char_limit,
        )
        trajectory_id = str(row["trajectory_id"])
        state_index = int(row["state_index"])
        hits.append(
            {
                "rank": rank,
                "source_id": int(row["source_id"]),
                "state_id": int(row["state_id"]),
                "trajectory_id": trajectory_id,
                "state_index": state_index,
                "sequence_ordinal": int(row["sequence_ordinal"]),
                "step": int(row["step"]),
                "exact_ref": (
                    f"trajectory://{uid}/{quote(trajectory_id, safe='')}/state/{state_index}"
                ),
                "match_kind": match_kind_by_id[int(row["state_id"])],
                "score": score_by_id[int(row["state_id"])],
                "goal": str(row["goal"]),
                "outcome": (
                    str(row["outcome"]) if row["outcome"] is not None else None
                ),
                "url": str(row["url"]),
                "incoming_action": (
                    str(row["incoming_action"])
                    if row["incoming_action"] is not None
                    else None
                ),
                "thoughts": (
                    str(row["thoughts"]) if row["thoughts"] is not None else None
                ),
                "text": text,
                "text_offset": text_offset,
                "text_truncated": text_truncated,
            }
        )
    return {
        "query": query,
        "scope_trajectory_count": len(source_ids),
        "semantic_status": semantic_status,
        "hits": hits,
    }


def load_span_rows(
    connection: sqlite3.Connection,
    *,
    trajectory_id: str,
    start_state_index: int,
    end_state_index: int,
    allowed_trajectory_ids: Sequence[str],
) -> list[sqlite3.Row]:
    require(
        trajectory_id in set(allowed_trajectory_ids),
        f"Trajectory is outside the question scope: {trajectory_id}",
    )
    rows = connection.execute(
        """
        SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
               a.relative_path, a.sha256 AS asset_sha256
        FROM lcm_trajectory_states s
        JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
        LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
        WHERE src.trajectory_id = ?
          AND s.state_index BETWEEN ? AND ?
        ORDER BY s.state_index
        """,
        (trajectory_id, int(start_state_index), int(end_state_index)),
    ).fetchall()
    expected = end_state_index - start_state_index + 1
    require(
        len(rows) == expected
        and [int(row["state_index"]) for row in rows]
        == list(range(start_state_index, end_state_index + 1)),
        (
            "Requested trajectory span does not resolve contiguously: "
            f"{trajectory_id} states {start_state_index}-{end_state_index}"
        ),
    )
    return rows


def render_search_result(result: dict[str, Any]) -> str:
    lines = [
        f"query: {result['query']}",
        f"scope_trajectories: {result['scope_trajectory_count']}",
        f"semantic_status: {result['semantic_status']}",
        f"hits: {len(result['hits'])}",
    ]
    for hit in result["hits"]:
        excerpt_label = "visible_state"
        if hit["text_truncated"]:
            excerpt_label = f"visible_state_excerpt(offset={hit['text_offset']})"
        lines.extend(
            [
                "",
                (
                    f"## Rank {hit['rank']} | {hit['match_kind']} | "
                    f"trajectory={hit['trajectory_id']} state={hit['state_index']} "
                    f"source_id={hit['source_id']} state_id={hit['state_id']}"
                ),
                f"exact_ref: {hit['exact_ref']}",
                f"goal: {hit['goal']}",
                f"outcome: {hit['outcome'] or '<unknown>'}",
                f"url: {hit['url']}",
                f"incoming_action: {hit['incoming_action'] or '<none>'}",
            ]
        )
        if hit["thoughts"]:
            lines.append(f"thoughts: {hit['thoughts']}")
        lines.extend([f"{excerpt_label}:", hit["text"]])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the current question's read-only Hermes-LCM trajectory scope."
    )
    parser.add_argument(
        "--scope",
        default=str(Path(__file__).with_name("hermes_scope.json")),
        help="Path to the module-generated question scope JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="Rank scoped state excerpts")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=None)
    search_parser.add_argument("--candidate-limit", type=int, default=None)
    search_parser.add_argument("--json", action="store_true")
    show_parser = subparsers.add_parser("show", help="Show one exact scoped state span")
    show_parser.add_argument("--trajectory-id", required=True)
    show_parser.add_argument("--start-state", type=int, required=True)
    show_parser.add_argument("--end-state", type=int, required=True)
    show_parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        scope = load_scope(Path(args.scope).expanduser().resolve())
        print(
            "[hermes_search] semantic self-check: "
            f"{json.dumps(semantic_self_check(scope))}",
            file=sys.stderr,
        )
        with connect_read_only(scope["canonical_store_path"]) as connection:
            resolve_scope_sources(connection, scope["trajectory_ids"])
            if args.command == "search":
                limit = min(
                    max(1, int(args.limit or scope.get("limit", 16))),
                    MAX_RESULTS,
                )
                candidate_limit = min(
                    max(
                        1,
                        int(
                            args.candidate_limit
                            or scope.get("candidate_limit", MAX_CANDIDATES)
                        ),
                    ),
                    MAX_CANDIDATES,
                )
                result = search_store(
                    connection,
                    args.query,
                    scope=scope,
                    limit=limit,
                    candidate_limit=candidate_limit,
                )
                if args.json:
                    print(json.dumps(result, indent=2, ensure_ascii=True))
                else:
                    print(render_search_result(result), end="")
                return 0

            require(
                args.start_state >= 0 and args.end_state >= args.start_state,
                "show requires 0 <= start-state <= end-state",
            )
            rows = load_span_rows(
                connection,
                trajectory_id=args.trajectory_id,
                start_state_index=args.start_state,
                end_state_index=args.end_state,
                allowed_trajectory_ids=scope["trajectory_ids"],
            )
            result = [
                {
                    "trajectory_id": str(row["trajectory_id"]),
                    "state_index": int(row["state_index"]),
                    "state_id": int(row["state_id"]),
                    "step": int(row["step"]),
                    "url": str(row["url"]),
                    "incoming_action": (
                        str(row["incoming_action"])
                        if row["incoming_action"] is not None
                        else None
                    ),
                    "thoughts": (
                        str(row["thoughts"]) if row["thoughts"] is not None else None
                    ),
                    "text": str(row["text"]),
                }
                for row in rows
            ]
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=True))
            else:
                for item in result:
                    print(
                        f"## trajectory={item['trajectory_id']} state={item['state_index']} "
                        f"state_id={item['state_id']} step={item['step']}\n"
                        f"url: {item['url']}\n"
                        f"incoming_action: {item['incoming_action'] or '<none>'}\n"
                        f"thoughts: {item['thoughts'] or '<none>'}\n"
                        f"visible_state:\n{item['text']}\n"
                    )
            return 0
    except Exception as exc:
        print(f"hermes_search error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
