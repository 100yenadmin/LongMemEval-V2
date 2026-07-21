#!/usr/bin/env python3
"""Freeze an answer-blind, balanced 30-question Hermes-LCM Small pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


OFFICIAL_HARNESS_COMMIT = "6f020ac2fc3275e46c706d3406e02c3ed79b7be2"
DEFAULT_SEED = "hermes-lcm-v2-small-pilot-v1"
DOMAINS = ("web", "enterprise")
STRATA = (
    ("static_state", "static-environment", 3),
    ("dynamic_state", "dynamic-environment", 3),
    ("workflow", "procedure", 3),
    ("environment_gotcha", "errors-gotchas", 3),
    ("premise_awareness", "static-environment-abs", 1),
    ("premise_awareness", "dynamic-environment-abs", 1),
    ("premise_awareness", "procedure-abs", 1),
)
SELECTION_INPUTS = (
    "id",
    "domain",
    "question_type",
    "image_presence",
    "small_haystack_membership",
)
FORBIDDEN_SELECTION_INPUTS = (
    "question",
    "answer",
    "eval_function",
    "references",
    "model_output",
    "judge_output",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"questions line {line_number} is not an object")
        rows.append(row)
    return rows


def _load_small_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Small haystack must be an object")
    small_ids: set[str] = set()
    for question_id, trajectory_ids in payload.items():
        if not isinstance(question_id, str) or not question_id:
            raise SystemExit("Small haystack contains an invalid question id")
        if not isinstance(trajectory_ids, list):
            raise SystemExit(f"Small haystack entry is not a list: {question_id}")
        small_ids.add(question_id)
    return small_ids


def _rank(seed: str, domain: str, ability: str, question_id: str) -> str:
    material = "\0".join((seed, domain, ability, question_id)).encode("utf-8")
    return _sha256_bytes(material)


def _candidate(
    row: dict[str, Any],
    *,
    seed: str,
    ability: str,
) -> dict[str, Any]:
    question_id = row.get("id")
    domain = row.get("domain")
    question_type = row.get("question_type")
    if not isinstance(question_id, str) or not question_id:
        raise SystemExit("eligible question has an invalid id")
    if domain not in DOMAINS:
        raise SystemExit(f"eligible question has invalid domain: {question_id}")
    if not isinstance(question_type, str) or not question_type:
        raise SystemExit(f"eligible question has invalid question_type: {question_id}")
    return {
        "ability": ability,
        "domain": domain,
        "has_image": isinstance(row.get("image"), str) and bool(row["image"].strip()),
        "question_id": question_id,
        "question_type": question_type,
        "rank_sha256": _rank(seed, domain, ability, question_id),
    }


def _promote_images(
    selected_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]],
    candidates_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    target_per_domain: int = 2,
) -> None:
    for domain in DOMAINS:
        while sum(
            bool(row["has_image"])
            for key, rows in selected_by_stratum.items()
            if key[0] == domain
            for row in rows
        ) < target_per_domain:
            replacements: list[
                tuple[str, tuple[str, str, str], dict[str, Any], dict[str, Any]]
            ] = []
            for key, selected in selected_by_stratum.items():
                if key[0] != domain:
                    continue
                selected_ids = {str(row["question_id"]) for row in selected}
                incoming = [
                    row
                    for row in candidates_by_stratum[key]
                    if row["has_image"] and row["question_id"] not in selected_ids
                ]
                outgoing = [row for row in selected if not row["has_image"]]
                if incoming and outgoing:
                    replacements.append((
                        str(incoming[0]["rank_sha256"]),
                        key,
                        incoming[0],
                        max(outgoing, key=lambda row: str(row["rank_sha256"])),
                    ))
            if not replacements:
                break
            _rank_value, key, incoming, outgoing = min(
                replacements,
                key=lambda item: (item[0], item[1]),
            )
            selected_by_stratum[key].remove(outgoing)
            selected_by_stratum[key].append(incoming)
            selected_by_stratum[key].sort(key=lambda row: str(row["rank_sha256"]))


def select_manifest(
    questions: list[dict[str, Any]],
    small_ids: set[str],
    *,
    seed: str,
) -> list[dict[str, Any]]:
    selected_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    candidates_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    eligible_ids: set[str] = set()
    for row in questions:
        question_id = row.get("id")
        if question_id not in small_ids:
            continue
        if not isinstance(question_id, str) or not question_id:
            raise SystemExit("eligible question has an invalid id")
        if question_id in eligible_ids:
            raise SystemExit(f"duplicate eligible question id: {question_id}")
        eligible_ids.add(question_id)
    seen_ids: set[str] = set()
    for domain in DOMAINS:
        for ability, question_type, quota in STRATA:
            key = (domain, ability, question_type)
            candidates = []
            for row in questions:
                if row.get("id") not in small_ids:
                    continue
                if row.get("domain") != domain or row.get("question_type") != question_type:
                    continue
                candidates.append(_candidate(row, seed=seed, ability=ability))
            candidates.sort(key=lambda row: str(row["rank_sha256"]))
            if len(candidates) < quota:
                raise SystemExit(
                    "quota deficit for "
                    f"domain={domain} ability={ability} question_type={question_type}: "
                    f"need {quota}, found {len(candidates)}"
                )
            candidates_by_stratum[key] = candidates
            selected_by_stratum[key] = list(candidates[:quota])

    _promote_images(selected_by_stratum, candidates_by_stratum)
    for domain in DOMAINS:
        image_count = sum(
            bool(row["has_image"])
            for key, rows in selected_by_stratum.items()
            if key[0] == domain
            for row in rows
        )
        if image_count < 2:
            raise SystemExit(
                f"image quota deficit for domain={domain}: need 2, found {image_count}"
            )
    selection: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for ability, question_type, _quota in STRATA:
            for row in selected_by_stratum[(domain, ability, question_type)]:
                question_id = str(row["question_id"])
                if question_id in seen_ids:
                    raise SystemExit(f"duplicate selected question id: {question_id}")
                seen_ids.add(question_id)
                selection.append(row)
    if len(selection) != 30:
        raise SystemExit(f"balanced manifest must contain 30 questions, found {len(selection)}")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the answer-blind Hermes-LCM LongMemEval-V2 Small pilot",
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--small-haystack", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    questions_path = Path(args.questions).expanduser().resolve()
    haystack_path = Path(args.small_haystack).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dataset_revision = str(args.dataset_revision).strip()
    seed = str(args.seed).strip()
    if not dataset_revision or not seed:
        raise SystemExit("dataset revision and seed must be non-empty")

    selection = select_manifest(
        _load_questions(questions_path),
        _load_small_ids(haystack_path),
        seed=seed,
    )
    manifest: dict[str, Any] = {
        "version": "hermes-lcm-v2-small-pilot-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": "https://github.com/xiaowu0162/LongMemEval-V2",
        "source_commit": OFFICIAL_HARNESS_COMMIT,
        "dataset_repo": "xiaowu0162/longmemeval-v2",
        "dataset_revision": dataset_revision,
        "tier": "small",
        "selection_seed": seed,
        "selection_rule": (
            "Exact per-domain question-type quotas; SHA-256(seed, domain, ability, id); "
            "then deterministic same-stratum image swaps targeting two images per domain."
        ),
        "selection_inputs": list(SELECTION_INPUTS),
        "forbidden_selection_inputs": list(FORBIDDEN_SELECTION_INPUTS),
        "selection_script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "source_questions_sha256": _sha256_bytes(questions_path.read_bytes()),
        "source_small_haystack_sha256": _sha256_bytes(haystack_path.read_bytes()),
        "selection": selection,
        "selection_digest": _sha256_json(selection),
        "proof_boundary": (
            "Predeclared diagnostic selection only; no retrieval, answer, judge, "
            "or official leaderboard claim."
        ),
    }
    manifest["manifest_digest"] = _sha256_json(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({
        "output": str(output_path),
        "questions": len(selection),
        "selection_digest": manifest["selection_digest"],
        "manifest_digest": manifest["manifest_digest"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
