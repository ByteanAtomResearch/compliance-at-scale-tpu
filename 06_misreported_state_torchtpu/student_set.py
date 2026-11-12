"""
Module 6: Build the student eval set from staged files.

The student track judges 22 records on one VM: the eight tamper records and
a fixed 14-record sample of the staged tier-2 batch. Both halves are staged
files, so nothing here waits on the Part 2 freeze or on anyone else's
adjudication.

    the eight   06_misreported_state_torchtpu/data/staging/candidates_tamper_batch1.jsonl
    the sample  06_misreported_state_torchtpu/data/staging/candidates_tier2_student14.jsonl

The one thing that is yours is the review of the eight. adjudicate.py writes
the records you promote to a corpus file and the ones you reject to a
rejections file; this script folds both back over the staged eight, in file
order, and appends the sample untouched. The output is one JSONL that every
later command reads, so the labels you set travel with the records and no
table has to be told where they came from.

What it refuses, with exit 2 and nothing written:
  - a tamper record you have not decided yet, unless you pass
    --allow-undecided (those records then say "proposed labels, not
    adjudicated" wherever labels are named);
  - a decision on a record that is not one of the eight;
  - a decision made against a different draft of a record. If the staged
    file changed after you reviewed it, your label is about a trajectory
    the judge will not see, and the record has to be reviewed again.

Usage:
    python 06_misreported_state_torchtpu/student_set.py \
        --decisions results/student/tamper_labels.jsonl \
        --rejections results/student/tamper_rejections.jsonl \
        --output results/student/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import lab05
from rich.console import Console

_schema = lab05.load("schema")
_corpus = lab05.load("corpus")
_labels = lab05.sibling("labels")
_review = lab05.sibling("build_review_files")

console = Console()

LAB06_DIR = Path(__file__).resolve().parent
TAMPER_FILENAME = "candidates_tamper_batch1.jsonl"
DEFAULT_TAMPER = LAB06_DIR / "data" / "staging" / TAMPER_FILENAME
DEFAULT_SAMPLE = LAB06_DIR / "data" / "staging" / _review.STUDENT14_FILENAME
DEFAULT_DECISIONS = "results/student/tamper_labels.jsonl"
DEFAULT_REJECTIONS = "results/student/tamper_rejections.jsonl"
DEFAULT_OUTPUT = "results/student/eval_set.jsonl"
EXIT_REFUSED = 2

# The fields adjudicate.py is allowed to change. Everything else is the
# trajectory the judge reads, and has to match the staged draft exactly.
REVIEW_FIELDS: tuple[str, ...] = ("labels", "adjudicated", "adjudication_note")


class Refusal(SystemExit):
    """The inputs were refused before anything was written. Carries the message for tests."""

    def __init__(self, message: str):
        super().__init__(EXIT_REFUSED)
        self.message = message


def refuse(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise Refusal(message)


def _trajectory(record: Any) -> dict[str, Any]:
    """A record with its review fields removed: what the judge is shown, plus the proposal."""
    doc = _schema.record_to_dict(record)
    for key in REVIEW_FIELDS:
        doc.pop(key, None)
    return doc


def load_rejections(path: str | Path) -> list:
    """Rejected records as adjudicate.py wrote them: staged, with the reason in the note."""
    records = _corpus.load_staged(path)
    not_rejections = [record.id for record in records if not _labels.is_rejected(record)]
    if not_rejections:
        refuse(f"{path}: these records carry no rejection note, so this is not a rejections file: {not_rejections}")
    return records


def merge(tamper: list, decided: list, rejected: list, allow_undecided: bool) -> tuple[list, dict[str, list[str]]]:
    """The eight in staged order, each replaced by your decision on it when you made one."""
    staged_by_id = {record.id: record for record in tamper}
    decisions = {record.id: record for record in decided}
    rejections = {record.id: record for record in rejected}

    problems: list[str] = []
    strangers = sorted((set(decisions) | set(rejections)) - set(staged_by_id))
    if strangers:
        problems.append(f"decisions on records that are not in the staged tamper file: {strangers}")
    both = sorted(set(decisions) & set(rejections))
    if both:
        problems.append(f"records both promoted and rejected: {both}")
    for record_id, record in {**rejections, **decisions}.items():
        staged = staged_by_id.get(record_id)
        if staged is not None and _trajectory(record) != _trajectory(staged):
            problems.append(
                f"{record_id}: the decision was made against a different draft of this record; "
                "delete it from your decisions file and review the record again"
            )
    undecided = [record.id for record in tamper if record.id not in decisions and record.id not in rejections]
    if undecided and not allow_undecided:
        problems.append(
            f"{len(undecided)} of {len(tamper)} tamper records have no decision yet: {undecided}. "
            "Run adjudicate.py again with the same paths (decided records are skipped), "
            "or pass --allow-undecided to build with their proposals, marked as not adjudicated"
        )
    if problems:
        refuse("refusing to build the student eval set:\n  " + "\n  ".join(problems))

    merged = [decisions.get(record.id) or rejections.get(record.id) or record for record in tamper]
    status = {
        "adjudicated": [record.id for record in tamper if record.id in decisions],
        "rejected": [record.id for record in tamper if record.id in rejections],
        "undecided": undecided,
    }
    return merged, status


def build(
    tamper_path: str | Path,
    sample_path: str | Path,
    decisions_path: str | Path,
    rejections_path: str | Path,
    out_path: str | Path,
    allow_undecided: bool = False,
) -> dict[str, Any]:
    """Write the student eval set and return what went into it."""
    tamper = _corpus.load_staged(tamper_path)
    sample = _corpus.load_staged(sample_path)
    if not tamper or not sample:
        refuse(f"need both staged files; loaded {len(tamper)} from {tamper_path} and {len(sample)} from {sample_path}")
    already = [record.id for record in tamper + sample if record.adjudicated]
    if already:
        refuse(f"staged inputs must be unadjudicated; these are not: {already}")
    try:
        decided = _corpus.load_corpus(decisions_path)
    except _corpus.CorpusIntegrityError as exc:
        refuse(f"{decisions_path} is not an adjudicate.py corpus file: {exc}")
    rejected = load_rejections(rejections_path)

    merged, status = merge(tamper, decided, rejected, allow_undecided)
    records = merged + sample
    for record in records:
        _schema.validate_record(record)
    duplicates = sorted(record_id for record_id, n in Counter(record.id for record in records).items() if n > 1)
    if duplicates:
        refuse(f"the eight and the sample share ids: {duplicates}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(_schema.record_to_dict(record), separators=(",", ":")) + "\n")
    return {
        "output": str(out_path),
        "records": len(records),
        "tamper": len(merged),
        "sample": len(sample),
        "status": status,
        "label_sources": _labels.summarize(records, staged=True),
    }


def main(args: argparse.Namespace) -> None:
    result = build(args.tamper, args.sample, args.decisions, args.rejections, args.output, args.allow_undecided)
    status = result["status"]
    console.print(
        f"The eight: {len(status['adjudicated'])} adjudicated, {len(status['rejected'])} rejected, "
        f"{len(status['undecided'])} undecided. The sample: {result['sample']} records, untouched.",
        highlight=False,
    )
    if status["undecided"]:
        console.print(
            f"[yellow]--allow-undecided: {status['undecided']} keep their proposals and are marked "
            f"'{_labels.LABELS_PROPOSED}'.[/yellow]",
            highlight=False,
            soft_wrap=True,
        )
    console.print(f"Labels: {result['label_sources']['statement']}", highlight=False, soft_wrap=True)
    console.print(
        f"[green]Student eval set written to {result['output']}: {result['records']} records.[/green]",
        highlight=False,
        soft_wrap=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 6: build the student eval set from staged files")
    parser.add_argument("--tamper", default=str(DEFAULT_TAMPER), help="The eight staged tamper records")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="The staged 14-record tier-2 sample")
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS, help="adjudicate.py --corpus file from your review")
    parser.add_argument("--rejections", default=DEFAULT_REJECTIONS, help="adjudicate.py --rejections file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where the eval set goes (results/ is gitignored)")
    parser.add_argument(
        "--allow-undecided",
        action="store_true",
        help="Build even when some of the eight have no decision; they are marked as not adjudicated",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
