"""
Module 5: Human adjudication of staged candidates.

This CLI is the only path into sample_data/trajectories.jsonl. It walks a
human through staged candidates one at a time; the human confirms, edits,
or rejects the proposed labels; only records marked adjudicated by that
human are appended to the scored corpus. promote_record() enforces the
invariant mechanically and tests/test_corpus.py holds it there.

Why this is not automatable: tier-2 candidates carry labels that are true
by construction, and the temptation is to bulk-promote them. The moment a
generator bug mislabels one scenario, every downstream precision number
inherits the bug silently. A human reading each record is the control, and
the adjudication note on every promoted record says who and when.

Decisions:
    a  accept the proposed labels as ground truth
    c  record is clean, no violations (overrides any proposal)
    e  edit: type labels as "state:step_index, state:step_index"
    r  reject: record goes to the rejections file with a reason, never
       to the corpus
    s  skip for now (no decision recorded)
    q  quit

Adjudication protocol (documented in the module README):
positives get a FULL review, every record, because they carry the labels
Table 3 is scored against. The templated clean negatives may instead be
spot-checked with --sample RATE: the tool walks a stratified random subset
(per clean sub-scenario, fixed seed), reports the disagreement rate found
in the sample, and only then offers to accept the unsampled remainder on
the template construction guarantee. Every record promoted that way says
so in its adjudication note, sample rate and disagreement count included,
so the protocol is enforced and visible in the corpus rather than left to
discipline.

Usage:
    # Full review (positives require this mode)
    python 05_trajectory_eval/adjudicate.py \
        --staged sample_data/staging/candidates_tier2_batch1.jsonl \
        --adjudicator noble

    # Spot-check the clean negatives at 20 percent
    python 05_trajectory_eval/adjudicate.py \
        --staged sample_data/staging/candidates_tier2_batch1.jsonl \
        --adjudicator noble --sample 0.2
"""

import argparse
import datetime
import json
import random
from pathlib import Path

from corpus import CORPUS_PATH, CorpusIntegrityError, load_corpus, load_staged
from rich.console import Console
from rich.table import Table
from schema import (
    VIOLATION_STATES,
    TrajectoryRecord,
    ViolationLabel,
    record_to_dict,
    validate_record,
)

console = Console()

REJECTIONS_PATH = "sample_data/staging/rejections.jsonl"


def promote_record(record: TrajectoryRecord, corpus_path: str | Path) -> None:
    """Append one adjudicated record to the scored corpus.

    This function is the corpus guardrail: it refuses unadjudicated records
    outright, and it is the only writer to the corpus in this module.
    """
    if not record.adjudicated:
        raise CorpusIntegrityError(
            f"record {record.id}: refusing to promote with adjudicated=false. "
            "Only human-adjudicated records enter the scored corpus."
        )
    validate_record(record)
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record_to_dict(record), separators=(",", ":")) + "\n")


def stratified_sample(records: list[TrajectoryRecord], rate: float, seed: int) -> list[TrajectoryRecord]:
    """Deterministic stratified sample of CLEAN candidates for spot-check.

    Strata are the clean sub-scenarios (the provenance template name), so
    every hard-negative shape gets sampled rather than whichever shape
    happens to sort first. Positives are excluded by construction: they are
    never spot-checked.
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"sample rate must be in (0, 1], got {rate}")
    clean = [r for r in records if not r.proposed_labels]
    strata: dict[str, list[TrajectoryRecord]] = {}
    for record in clean:
        strata.setdefault(record.provenance, []).append(record)
    rng = random.Random(seed)
    sampled: list[TrajectoryRecord] = []
    for name in sorted(strata):
        members = strata[name]
        take = max(1, round(rate * len(members)))
        sampled.extend(rng.sample(members, min(take, len(members))))
    return sampled


def accept_remainder_on_construction(
    remainder: list[TrajectoryRecord],
    corpus_path: str | Path,
    adjudicator: str,
    rate: float,
    sampled_n: int,
    disagreements: int,
) -> int:
    """Promote unsampled clean candidates with an adjudication note that
    records the accept-on-construction decision, the spot-check rate, and
    the disagreement count, so the corpus itself discloses the protocol."""
    stamp = f"{adjudicator} {datetime.date.today().isoformat()}"
    promoted = 0
    for record in remainder:
        record.labels = []
        record.adjudicated = True
        record.adjudication_note = (
            f"{stamp} accepted on template construction guarantee "
            f"(spot-check rate {rate:.0%}, sampled {sampled_n}, disagreements {disagreements})"
        )
        promote_record(record, corpus_path)
        promoted += 1
    return promoted


def parse_label_edit(text: str, valid_indices: set[int]) -> list[ViolationLabel]:
    """Parse "state:step_index, state:step_index" into labels, strictly."""
    labels = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        state, _, idx = part.partition(":")
        state = state.strip()
        if state not in VIOLATION_STATES:
            raise ValueError(f"unknown state {state!r} (valid: {', '.join(VIOLATION_STATES)})")
        try:
            step_index = int(idx)
        except ValueError as exc:
            raise ValueError(f"bad step index {idx!r} for {state}") from exc
        if step_index not in valid_indices:
            raise ValueError(f"step {step_index} does not exist on this record")
        labels.append(ViolationLabel(state=state, step_index=step_index))
    if not labels:
        raise ValueError("no labels parsed; use 'state:step_index' or choose [c] for clean")
    return labels


def _show(record: TrajectoryRecord) -> None:
    console.print()
    console.rule(f"[bold]{record.id}[/bold]  tier={record.tier}  provenance={record.provenance}")
    console.print(f"[bold]Goal:[/bold] {record.goal}")
    for b in record.boundaries:
        console.print(f"  boundary: {b}")

    table = Table(show_lines=False)
    for col in ("#", "actor", "action", "target", "approval", "status", "signals", "agent_report"):
        table.add_column(col, overflow="fold")
    for s in record.steps:
        table.add_row(
            str(s.index),
            s.actor,
            s.action,
            s.target,
            s.approval_state,
            s.status,
            ",".join(s.stop_signals),
            s.agent_report[:80],
        )
    for s in record.overflow_steps:
        table.add_row(
            f"{s.index}*",
            s.actor,
            s.action,
            s.target,
            s.approval_state,
            s.status,
            ",".join(s.stop_signals),
            s.agent_report[:80],
        )
    console.print(table)
    if record.truncation:
        console.print(
            f"[dim]* reduced steps (strategy {record.truncation.strategy}, "
            f"dropped: {', '.join(record.truncation.dropped_fields)})[/dim]"
        )
    console.print(f"[bold]Final report:[/bold] {record.final_report}")
    if record.proposed_labels:
        proposed = ", ".join(f"{lb.state}@{lb.step_index}" for lb in record.proposed_labels)
        console.print(f"[yellow]Proposed labels: {proposed}[/yellow]")
    else:
        console.print("[yellow]Proposed: clean (no violations)[/yellow]")


def _decide(record: TrajectoryRecord, adjudicator: str) -> tuple[str, TrajectoryRecord | None]:
    """One decision loop for one record. Returns (action, record_or_none)."""
    valid_indices = {s.index for s in record.steps} | {s.index for s in record.overflow_steps}
    stamp = f"{adjudicator} {datetime.date.today().isoformat()}"
    while True:
        try:
            choice = console.input("[a]ccept  [c]lean  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit", None
        if choice == "a":
            if not record.proposed_labels:
                console.print("[red]No proposed labels to accept; use [c] for clean or [e] to add labels.[/red]")
                continue
            record.labels = list(record.proposed_labels)
            record.adjudicated = True
            record.adjudication_note = f"{stamp} accepted proposal"
            return "promote", record
        if choice == "c":
            record.labels = []
            record.adjudicated = True
            record.adjudication_note = f"{stamp} adjudicated clean"
            return "promote", record
        if choice == "e":
            try:
                raw = console.input("labels (state:step_index, ...) > ")
                record.labels = parse_label_edit(raw, valid_indices)
            except (EOFError, KeyboardInterrupt):
                return "quit", None
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            record.adjudicated = True
            record.adjudication_note = f"{stamp} edited labels"
            return "promote", record
        if choice == "r":
            try:
                reason = console.input("rejection reason > ").strip()
            except (EOFError, KeyboardInterrupt):
                return "quit", None
            record.adjudication_note = f"{stamp} rejected: {reason}"
            return "reject", record
        if choice == "s":
            return "skip", None
        if choice == "q":
            return "quit", None
        console.print("[red]Unrecognized choice.[/red]")


def main(args: argparse.Namespace) -> None:
    staged = load_staged(args.staged)
    corpus = load_corpus(args.corpus)
    done_ids = {r.id for r in corpus}

    rejections_path = Path(args.rejections)
    if rejections_path.exists():
        with open(rejections_path, encoding="utf-8") as f:
            done_ids |= {json.loads(line)["id"] for line in f if line.strip()}

    pending = [r for r in staged if r.id not in done_ids][args.start :]
    console.print(f"{len(pending)} candidates pending ({len(staged)} staged, {len(done_ids)} already decided)")

    if args.sample:
        review_list = stratified_sample(pending, args.sample, args.seed)
        positives = sum(1 for r in pending if r.proposed_labels)
        console.print(
            f"[bold]Spot-check mode:[/bold] {len(review_list)} clean candidates sampled at {args.sample:.0%} "
            f"(stratified by clean sub-scenario, seed {args.seed}). "
            f"{positives} positives are excluded; they require a full-review session."
        )
    else:
        review_list = pending

    promoted = rejected = skipped = disagreements = 0
    quit_early = False
    for record in review_list:
        _show(record)
        action, decided = _decide(record, args.adjudicator)
        if action == "promote":
            promote_record(decided, args.corpus)
            promoted += 1
            if args.sample and decided.labels:
                # A sampled "clean" candidate that turned out to carry a
                # violation is exactly what the spot-check exists to find.
                disagreements += 1
            console.print(f"[green]Promoted {decided.id} ({promoted} this session).[/green]")
        elif action == "reject":
            rejections_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rejections_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record_to_dict(decided), separators=(",", ":")) + "\n")
            rejected += 1
            if args.sample:
                disagreements += 1
            console.print(f"[yellow]Rejected {decided.id}.[/yellow]")
        elif action == "skip":
            skipped += 1
        elif action == "quit":
            quit_early = True
            break

    console.print(f"Session done: {promoted} promoted, {rejected} rejected. Corpus: {args.corpus}")

    if args.sample and not quit_early and skipped == 0:
        sample_ids = {r.id for r in review_list}
        remainder = [r for r in pending if not r.proposed_labels and r.id not in sample_ids]
        console.print(
            f"Sample complete: {disagreements} disagreement(s) in {len(review_list)} sampled clean candidates."
        )
        if remainder:
            try:
                answer = console.input(
                    f"Accept the remaining {len(remainder)} clean candidates on the template construction "
                    f"guarantee? The decision and the sample stats go into every adjudication note. [y/N] > "
                )
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer.strip().lower() == "y":
                n = accept_remainder_on_construction(
                    remainder, args.corpus, args.adjudicator, args.sample, len(review_list), disagreements
                )
                console.print(f"[green]{n} clean candidates accepted on construction guarantee.[/green]")
            else:
                console.print("Remainder left staged; re-run to review or sample again.")
    elif args.sample and (quit_early or skipped):
        console.print("[yellow]Sample incomplete (quit or skips); remainder acceptance not offered.[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjudicate staged trajectory candidates into the scored corpus")
    parser.add_argument("--staged", required=True, help="Staged candidate JSONL to review")
    parser.add_argument("--corpus", default=CORPUS_PATH, help="Scored corpus JSONL to append to")
    parser.add_argument("--rejections", default=REJECTIONS_PATH, help="Where rejected candidates are recorded")
    parser.add_argument("--adjudicator", required=True, help="Name recorded in every adjudication note")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N pending candidates")
    parser.add_argument("--sample", type=float, default=None, help="Spot-check clean negatives at this rate (0-1)")
    parser.add_argument("--seed", type=int, default=20260812, help="Sampling seed, fixed so the subset is auditable")
    main(parser.parse_args())
