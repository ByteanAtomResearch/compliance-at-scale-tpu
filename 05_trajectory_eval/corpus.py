"""
Module 5: Corpus loading, validation, and class-balance reporting.

The scored corpus lives at sample_data/trajectories.jsonl and holds only
records a human adjudicated. That boundary is what makes Table 3 mean
anything: a judge scored against labels nobody verified is measuring
agreement with a template, not detection of violations. The loader here
enforces the boundary mechanically, and tests/test_corpus.py exists to keep
it enforced.

Candidates wait in sample_data/staging/ (committed to the repo, so the diff
between what was proposed and what was promoted stays reviewable). The only
path from staging to the corpus is adjudicate.py.

Usage:
    # Balance report for the scored corpus (empty corpus prints all-zero rows)
    python 05_trajectory_eval/corpus.py

    # Preview a staged candidate file against the same targets
    python 05_trajectory_eval/corpus.py --staged sample_data/staging/candidates_tier2_batch1.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from schema import TIERS, VIOLATION_STATES, TrajectoryRecord, record_from_dict, validate_record

console = Console()

CORPUS_PATH = "sample_data/trajectories.jsonl"

# Target shape from the run plan: enough positives per state for per-state
# recall to mean something (with intervals reported honestly), plus a clean
# majority so precision is exercised.
TARGET_POSITIVES_PER_STATE = 25
TARGET_CLEAN_NEGATIVES = 100


class CorpusIntegrityError(ValueError):
    """A record that must not be in the scored corpus is in it."""


def load_records(path: str | Path, require_adjudicated: bool) -> list[TrajectoryRecord]:
    """Load and validate a JSONL file of trajectory records.

    require_adjudicated=True is the scored-corpus mode: one unadjudicated
    record anywhere in the file fails the whole load, loudly. There is no
    skip-and-continue here because a partially trusted corpus is worse than
    a missing one.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[TrajectoryRecord] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = record_from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise CorpusIntegrityError(f"{path} line {line_num}: unreadable record ({exc})") from exc
            validate_record(record)
            if require_adjudicated and not record.adjudicated:
                raise CorpusIntegrityError(
                    f"{path} line {line_num}: record {record.id} has adjudicated=false. "
                    "Unadjudicated records never enter the scored corpus; promote through adjudicate.py."
                )
            records.append(record)
    _reject_duplicate_ids(records, path)
    return records


def load_corpus(path: str | Path = CORPUS_PATH) -> list[TrajectoryRecord]:
    """The scored corpus: adjudicated records only, or an empty list."""
    return load_records(path, require_adjudicated=True)


def load_staged(path: str | Path) -> list[TrajectoryRecord]:
    """A staged candidate file: proposed labels, adjudicated=false expected."""
    return load_records(path, require_adjudicated=False)


def _reject_duplicate_ids(records: list[TrajectoryRecord], path: Path) -> None:
    counts = Counter(r.id for r in records)
    dupes = [rid for rid, n in counts.items() if n > 1]
    if dupes:
        raise CorpusIntegrityError(f"{path}: duplicate record ids {dupes}")


def class_balance(records: list[TrajectoryRecord], label_source: str = "labels") -> dict[str, Any]:
    """Count positives per state (split by tier, variant, and region) and
    clean negatives.

    label_source is "labels" for the scored corpus and "proposed" for a
    staging preview. The two are never mixed in one report, because proposed
    counts describe intent and label counts describe ground truth. Region
    counts matter because only full_fidelity positives feed the headline
    results; reduced_region positives feed the instrumentation-dependency
    table, and a state whose positives are mostly reduced would be
    under-filled where it counts.
    """
    if label_source not in ("labels", "proposed"):
        raise ValueError(f"label_source must be 'labels' or 'proposed', got {label_source!r}")

    per_state: dict[str, dict[str, Any]] = {
        state: {"total": 0, "by_tier": dict.fromkeys(TIERS, 0), "by_variant": Counter(), "by_region": Counter()}
        for state in VIOLATION_STATES
    }
    clean = 0
    for record in records:
        labels = record.labels if label_source == "labels" else record.proposed_labels
        states = {label.state for label in labels}
        if not states:
            clean += 1
            continue
        for label in labels:
            per_state[label.state]["by_region"][label.region] += 1
        for state in states:
            per_state[state]["total"] += 1
            per_state[state]["by_tier"][record.tier] += 1
            if record.variant != "na":
                per_state[state]["by_variant"][record.variant] += 1

    return {
        "total_records": len(records),
        "per_state": per_state,
        "clean_negatives": clean,
        "under_filled": [s for s in VIOLATION_STATES if per_state[s]["total"] < TARGET_POSITIVES_PER_STATE],
        "clean_shortfall": max(0, TARGET_CLEAN_NEGATIVES - clean),
    }


def print_balance(balance: dict[str, Any], title: str) -> None:
    """Class-balance table with targets and shortfall flags."""
    table = Table(title=title)
    table.add_column("Violation state", style="cyan")
    table.add_column("Positives", justify="right")
    table.add_column("Target", justify="right")
    for tier in TIERS:
        table.add_column(tier, justify="right", style="dim")
    table.add_column("Variants", style="dim")
    table.add_column("Regions", style="dim")
    table.add_column("Status")

    for state, stats in balance["per_state"].items():
        variants = ", ".join(f"{k}={v}" for k, v in sorted(stats["by_variant"].items())) or "-"
        regions = ", ".join(f"{k}={v}" for k, v in sorted(stats["by_region"].items())) or "-"
        short = stats["total"] < TARGET_POSITIVES_PER_STATE
        table.add_row(
            state,
            str(stats["total"]),
            str(TARGET_POSITIVES_PER_STATE),
            *(str(stats["by_tier"][tier]) for tier in TIERS),
            variants,
            regions,
            "[red]UNDER-FILLED[/red]" if short else "[green]ok[/green]",
        )

    clean = balance["clean_negatives"]
    table.add_row(
        "[bold]clean negatives[/bold]",
        str(clean),
        str(TARGET_CLEAN_NEGATIVES),
        *("-" for _ in TIERS),
        "-",
        "-",
        "[red]UNDER-FILLED[/red]" if clean < TARGET_CLEAN_NEGATIVES else "[green]ok[/green]",
    )

    console.print(table)
    console.print(f"Total records: {balance['total_records']}")
    if balance["under_filled"]:
        console.print(f"[yellow]Under-filled states: {', '.join(balance['under_filled'])}[/yellow]")


def _step_count(record: TrajectoryRecord) -> int:
    return len(record.steps) + len(record.overflow_steps)


def print_length_summary(records: list[TrajectoryRecord]) -> None:
    """Step-count spread, the input to Phase 4 band selection."""
    if not records:
        return
    counts = sorted(_step_count(r) for r in records)
    truncated = sum(1 for r in records if r.truncation is not None)
    table = Table(title="Step-count distribution")
    table.add_column("min", justify="right")
    table.add_column("median", justify="right")
    table.add_column("max", justify="right")
    table.add_column("records past cap", justify="right")
    table.add_row(str(counts[0]), str(counts[len(counts) // 2]), str(counts[-1]), str(truncated))
    console.print(table)


def main(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.corpus)
    print_balance(class_balance(corpus, label_source="labels"), f"Scored corpus: {args.corpus}")
    print_length_summary(corpus)

    if args.staged:
        console.print()
        staged = load_staged(args.staged)
        unadjudicated = sum(1 for r in staged if not r.adjudicated)
        console.print(f"Staged candidates: {len(staged)} ({unadjudicated} awaiting adjudication)")
        print_balance(class_balance(staged, label_source="proposed"), f"Staged (proposed labels): {args.staged}")
        print_length_summary(staged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-balance report for the trajectory corpus")
    parser.add_argument("--corpus", default=CORPUS_PATH, help="Path to the scored corpus JSONL")
    parser.add_argument("--staged", default=None, help="Optional staged candidate file to preview")
    main(parser.parse_args())
