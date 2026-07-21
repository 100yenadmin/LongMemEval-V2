"""Provider-free contract for the predeclared balanced Small pilot manifest."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "evaluation" / "scripts" / "select_hermes_small_pilot.py"


def _fixture_rows(*, poison: str = "first") -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    haystack: dict[str, list[str]] = {}
    question_types = (
        "static-environment",
        "dynamic-environment",
        "procedure",
        "errors-gotchas",
        "static-environment-abs",
        "dynamic-environment-abs",
        "procedure-abs",
    )
    for domain in ("web", "enterprise"):
        for question_type in question_types:
            for index in range(6):
                question_id = f"{domain}-{question_type}-{index}"
                rows.append({
                    "id": question_id,
                    "domain": domain,
                    "question_type": question_type,
                    "image": f"images/{question_id}.png" if index in {4, 5} else None,
                    "question": f"poison question {poison} {question_id}",
                    "answer": f"poison answer {poison} {question_id}",
                    "eval_function": f"poison judge {poison}",
                    "references": [f"poison reference {poison}"],
                })
                haystack[question_id] = [f"trajectory-{domain}-{index}"]
    return rows, haystack


def _write_inputs(root: Path, *, poison: str = "first") -> tuple[Path, Path]:
    rows, haystack = _fixture_rows(poison=poison)
    questions = root / "questions.jsonl"
    questions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    haystack_path = root / "lme_v2_small.json"
    haystack_path.write_text(json.dumps(haystack, sort_keys=True) + "\n", encoding="utf-8")
    return questions, haystack_path


def _run_selector(root: Path, *, poison: str = "first") -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    questions, haystack = _write_inputs(root, poison=poison)
    output = root / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions",
            str(questions),
            "--small-haystack",
            str(haystack),
            "--dataset-revision",
            "dataset-revision-1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_manifest_is_balanced_deterministic_and_answer_blind(tmp_path: Path):
    first = _run_selector(tmp_path / "first", poison="first")
    second = _run_selector(tmp_path / "second", poison="changed")

    assert first["selection"] == second["selection"]
    assert first["selection_digest"] == second["selection_digest"]
    assert len(first["selection"]) == 30
    counts = Counter(
        (row["domain"], row["ability"])
        for row in first["selection"]
    )
    assert set(counts.values()) == {3}
    image_counts = Counter(
        row["domain"] for row in first["selection"] if row["has_image"]
    )
    assert image_counts["web"] >= 2
    assert image_counts["enterprise"] >= 2
    assert set(first["selection_inputs"]) == {
        "id",
        "domain",
        "question_type",
        "image_presence",
        "small_haystack_membership",
    }
    assert set(first["forbidden_selection_inputs"]) == {
        "question",
        "answer",
        "eval_function",
        "references",
        "model_output",
        "judge_output",
    }
    assert all(
        set(row) == {
            "ability",
            "domain",
            "has_image",
            "question_id",
            "question_type",
            "rank_sha256",
        }
        for row in first["selection"]
    )
    serialized = json.dumps(first, sort_keys=True)
    assert "poison answer" not in serialized
    assert "poison reference" not in serialized
    assert first["manifest_digest"]
    assert first["selection_script_sha256"]


def test_manifest_fails_closed_on_exact_quota_deficit(tmp_path: Path):
    questions, haystack_path = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in questions.read_text().splitlines()]
    rows = [
        row
        for row in rows
        if not (
            row["domain"] == "enterprise"
            and row["question_type"] == "errors-gotchas"
        )
    ]
    questions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions",
            str(questions),
            "--small-haystack",
            str(haystack_path),
            "--dataset-revision",
            "dataset-revision-1",
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "quota deficit" in completed.stderr
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_rejects_duplicate_eligible_ids(tmp_path: Path):
    questions, haystack_path = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in questions.read_text().splitlines()]
    duplicate = dict(rows[0])
    duplicate["answer"] = "different poison answer"
    rows.append(duplicate)
    questions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions",
            str(questions),
            "--small-haystack",
            str(haystack_path),
            "--dataset-revision",
            "dataset-revision-1",
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "duplicate eligible question id" in completed.stderr
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_fails_when_per_domain_image_target_is_unavailable(tmp_path: Path):
    questions, haystack_path = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in questions.read_text().splitlines()]
    for row in rows:
        if row["domain"] == "web":
            row["image"] = None
    questions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions",
            str(questions),
            "--small-haystack",
            str(haystack_path),
            "--dataset-revision",
            "dataset-revision-1",
            "--output",
            str(tmp_path / "manifest.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "image quota deficit for domain=web" in completed.stderr
    assert not (tmp_path / "manifest.json").exists()
