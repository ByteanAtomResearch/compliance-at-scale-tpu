"""
Module 6: Three readers of the same records, case by case.

The rules baseline, the judge and a set of labels each say which violation
states a trajectory carries and at which step. This script lines the three
up, one row per record, so you can see where they part ways. It reads:

    --records  the labelled records (the student eval set, with --allow-staged)
    --rules    a verdict file from 05's rules_baseline.py over those records
    --judge    a verdict file from run.py --eval-set over those records

It prints one table per tier and names, in each table's title, whose labels
that tier carries (labels.py). The eight tamper records and the 14-record
sample land in different tiers, so their labels are never mixed in a table.

Counts only. Every figure here is a count of records, never a rate: with
eight records, one record is an eighth of any rate you could compute. The
JSON this script writes holds no rate either, so none can be lifted out of
it later.

It refuses, with exit 2 and nothing written, when the three readers did not
read the same records: a record with no rules verdict or no judge verdict,
or a verdict for a record that is not in the file. A record rejected at
adjudication has no label, so it is shown with what the rules and the judge
said and counted on neither side.

Usage:
    python 06_misreported_state_torchtpu/compare_readers.py \
        --records results/student/eval_set.jsonl --allow-staged \
        --rules results/student/rules.json \
        --judge results/student/verdicts_torchtpu.json \
        --output results/student/readers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import lab05
from rich.console import Console
from rich.table import Table

_schema = lab05.load("schema")
_corpus = lab05.load("corpus")
_labels = lab05.sibling("labels")

# Piped through tee, rich assumes 80 columns and would crop the per-case table.
PIPED_WIDTH = 132
console = Console()
if not console.is_terminal:
    console = Console(width=PIPED_WIDTH)

EXIT_REFUSED = 2
COUNTS_NOTE = "per-case comparison; every figure is a count of records, never a rate"
UNLOCALIZED = -1


class Refusal(SystemExit):
    """The inputs were refused before anything was written. Carries the message for tests."""

    def __init__(self, message: str):
        super().__init__(EXIT_REFUSED)
        self.message = message


def refuse(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise Refusal(message)


# ── Loading ───────────────────────────────────────────────────────────────────


def load_verdict_file(path: str | Path, flag: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if "verdicts" not in doc:
        raise SystemExit(f"{path}: no 'verdicts' key; {flag} needs a verdict file")
    return doc


def load_records(path: str | Path, allow_staged: bool) -> list:
    try:
        records = _corpus.load_records(path, require_adjudicated=not allow_staged)
    except _corpus.CorpusIntegrityError as exc:
        raise SystemExit(f"{exc}\nPass --allow-staged to read a file that is not adjudicated throughout.") from exc
    if not records:
        raise SystemExit(f"no records loaded from {path}")
    return records


def same_records(records: list, rules: dict[str, Any], judge: dict[str, Any]) -> None:
    """All three readers, or none of them. A missing verdict would read as 'clean' and that would be a lie."""
    ids = {record.id for record in records}
    problems: list[str] = []
    for name, verdicts in (("rules", rules), ("judge", judge)):
        missing, extra = sorted(ids - set(verdicts)), sorted(set(verdicts) - ids)
        if missing:
            problems.append(f"{len(missing)} records have no {name} verdict: {missing[:20]}")
        if extra:
            problems.append(f"{len(extra)} {name} verdicts are for records not in the file: {extra[:20]}")
    if problems:
        refuse("refusing to compare: the three readers did not read the same records\n  " + "\n  ".join(problems))


# ── One record ────────────────────────────────────────────────────────────────


def _findings(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """A reader's findings as state and step, in 05's state order."""
    order = {state: index for index, state in enumerate(_schema.VIOLATION_STATES)}
    found = [
        {"state": violation["state"], "step": int(violation.get("failed_step_index", UNLOCALIZED))}
        for violation in verdict.get("violations", [])
    ]
    return sorted(found, key=lambda item: (order.get(item["state"], len(order)), item["step"]))


def _against_labels(labels: list[dict[str, Any]] | None, found: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Where a reader and the labels part ways, by state, then by step where the state matches."""
    if labels is None:
        return None
    label_states = {item["state"] for item in labels}
    found_states = {item["state"] for item in found}
    order = _schema.VIOLATION_STATES
    step_differs = []
    for state in order:
        if state in label_states and state in found_states:
            label_steps = sorted(item["step"] for item in labels if item["state"] == state)
            found_steps = sorted(item["step"] for item in found if item["state"] == state)
            if label_steps != found_steps:
                step_differs.append({"state": state, "label_steps": label_steps, "reader_steps": found_steps})
    return {
        "same_states": label_states == found_states,
        "missed": [state for state in order if state in label_states - found_states],
        "extra": [state for state in order if state in found_states - label_states],
        "step_differs": step_differs,
    }


def case(record: Any, rules: dict[str, Any], judge: dict[str, Any], staged: bool) -> dict[str, Any]:
    usable = _labels.scoring_labels(record)
    labels = None if usable is None else [{"state": label.state, "step": label.step_index} for label in usable]
    rules_found, judge_found = _findings(rules[record.id]), _findings(judge[record.id])
    return {
        "id": record.id,
        "tier": record.tier,
        "draft": record.provenance.rsplit("/", 1)[-1],
        "label_source": _labels.label_source(record, staged),
        "labels": labels,
        "rules": rules_found,
        "judge": judge_found,
        "rules_vs_labels": _against_labels(labels, rules_found),
        "judge_vs_labels": _against_labels(labels, judge_found),
    }


def tier_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    labelled = [row for row in cases if row["labels"] is not None]
    return {
        "records": len(cases),
        "with_label": len(labelled),
        "labelled_with_a_violation": sum(1 for row in labelled if row["labels"]),
        "labelled_clean": sum(1 for row in labelled if not row["labels"]),
        "rules_same_states_as_labels": sum(1 for row in labelled if row["rules_vs_labels"]["same_states"]),
        "judge_same_states_as_labels": sum(1 for row in labelled if row["judge_vs_labels"]["same_states"]),
        "rules_flagged_any": sum(1 for row in cases if row["rules"]),
        "judge_flagged_any": sum(1 for row in cases if row["judge"]),
    }


# ── Comparison ────────────────────────────────────────────────────────────────


def _reader_info(path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    git = metadata.get("git")
    sources = metadata.get("label_sources")
    return {
        "path": str(path),
        "evaluator": metadata.get("evaluator"),
        "backend": metadata.get("backend"),
        "model": metadata.get("model"),
        "parse_errors": metadata.get("parse_errors"),
        "git_sha": git.get("sha") if isinstance(git, dict) else None,
        "label_sources_at_run": sources.get("statement") if isinstance(sources, dict) else None,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.records, args.allow_staged)
    rules_doc, judge_doc = load_verdict_file(args.rules, "--rules"), load_verdict_file(args.judge, "--judge")
    rules, judge = rules_doc["verdicts"], judge_doc["verdicts"]
    same_records(records, rules, judge)

    sources = _labels.summarize(records, staged=args.allow_staged)
    cases = [case(record, rules, judge, args.allow_staged) for record in records]
    tiers = [tier for tier in _schema.TIERS if tier in sources["by_tier"]]
    return {
        "metadata": {
            "tool": "compare_readers",
            "note": COUNTS_NOTE,
            "records": str(args.records),
            "label_sources": sources,
            "rules": _reader_info(args.rules, rules_doc.get("metadata", {})),
            "judge": _reader_info(args.judge, judge_doc.get("metadata", {})),
        },
        "counts": {tier: tier_counts([row for row in cases if row["tier"] == tier]) for tier in tiers},
        "cases": cases,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def _cell(found: list[dict[str, Any]] | None) -> str:
    if found is None:
        return "rejected, no label"
    if not found:
        return "clean"
    return "\n".join(f"{item['state']}@{'?' if item['step'] == UNLOCALIZED else item['step']}" for item in found)


def _difference(versus: dict[str, Any] | None) -> str:
    if versus is None:
        return "-"
    parts = [f"missed {state}" for state in versus["missed"]] + [f"extra {state}" for state in versus["extra"]]
    for item in versus["step_differs"]:
        parts.append(f"{item['state']}: step {item['reader_steps']}, label {item['label_steps']}")
    return "\n".join(parts) if parts else "same"


def print_report(result: dict[str, Any]) -> None:
    sources = result["metadata"]["label_sources"]
    for tier, counts in result["counts"].items():
        whose = _labels.tier_labels(sources, tier)
        title = f"Three readers, tier {tier}, {counts['records']} records. Labels: {whose}"
        table = Table(title=title, show_lines=True)
        table.add_column("Record", no_wrap=True)
        for column in ("Labels", "Rules", "Judge", "Judge vs labels"):
            table.add_column(column, overflow="fold")
        for row in result["cases"]:
            if row["tier"] != tier:
                continue
            table.add_row(
                row["id"],
                _cell(row["labels"]),
                _cell(row["rules"]),
                _cell(row["judge"]),
                _difference(row["judge_vs_labels"]),
            )
        console.print(table)
        console.print(
            f"  Of {counts['with_label']} labelled records ({counts['labelled_with_a_violation']} with a violation, "
            f"{counts['labelled_clean']} clean): the rules name the same states as the labels on "
            f"{counts['rules_same_states_as_labels']}, the judge on {counts['judge_same_states_as_labels']}. "
            "Counts, not rates.",
            highlight=False,
            soft_wrap=True,
        )
    judge = result["metadata"]["judge"]
    at_run = judge["label_sources_at_run"]
    if at_run is not None and at_run != sources["statement"]:
        console.print(
            "[yellow]The judge run recorded different label sources than this file carries. Labels never reach "
            f"the judge, so the verdicts stand, but check this is the file that run read.\n  at run:  {at_run}\n"
            f"  in file: {sources['statement']}[/yellow]",
            highlight=False,
            soft_wrap=True,
        )
    console.print(
        f"  Judge: {judge['evaluator']} on backend {judge['backend']}, parse_errors={judge['parse_errors']}, "
        f"commit {judge['git_sha']}.",
        highlight=False,
        soft_wrap=True,
    )


# ── Entry ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    result = compare(args)
    print_report(result)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        console.print(f"[green]Reader comparison written to {out_path}[/green]", highlight=False, soft_wrap=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 6: rules, judge and labels, one row per record")
    parser.add_argument("--records", required=True, help="Labelled records all three readers read")
    parser.add_argument(
        "--allow-staged",
        action="store_true",
        help="Accept a file that is not adjudicated throughout; each table says whose labels it shows",
    )
    parser.add_argument("--rules", required=True, help="Verdict file from 05's rules_baseline.py")
    parser.add_argument("--judge", required=True, help="Verdict file from run.py --eval-set")
    parser.add_argument("--output", default=None, help="Where to write the per-case JSON; nothing on refusal")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
