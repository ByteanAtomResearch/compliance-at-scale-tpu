"""
Module 5: Token-length distribution and band selection.

XLA compiles one graph per batch shape, at 20 to 30 minutes per shape on
v5e-4. Bucketing prompts into length bands trades compile count against
padding waste: one band pads every short prompt to the longest, many bands
compile many graphs. The right boundaries come out of the observed token
distribution of the corpus, never out of round numbers, and this script is
where they come from.

What gets tokenized: RENDERED judge prompts, not raw records. Prompt
scaffolding (framing, definitions, instructions) contributes tokens to
every band, and a band chosen from record length alone would be wrong by
exactly that scaffold. Each record's length is the maximum over its six
per-state prompts, because a record's band has to fit whichever of its six
calls is longest.

Band ceilings are rounded up to multiples of 64 tokens (XLA-friendly
shapes). The proposal rule is stated in the output and applied
deterministically: prefer the smallest band count where adding one more
band stops saving at least 5 percent of padded tokens, within the 3-to-4
band budget the compile calendar allows.

The Gemma checkpoint is gated on Hugging Face, so running this against the
real tokenizer needs HF auth (huggingface-cli login or HF_TOKEN). That
friction is logged in the E4 friction log rather than worked around.

Usage:
    uv run python 05_trajectory_eval/bands.py --corpus sample_data/trajectories.jsonl
    uv run python 05_trajectory_eval/bands.py --corpus sample_data/staging/candidates_tier2_batch1.jsonl --staged
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from corpus import load_corpus, load_staged
from prompts import PROMPT_VERSION, prompt_hash, render_state_prompt
from rich.console import Console
from rich.table import Table
from schema import VIOLATION_STATES, TrajectoryRecord

console = Console()

SHAPE_MULTIPLE = 64
BAND_COUNTS = (1, 2, 3, 4)
MARGINAL_WASTE_CUTOFF = 0.05  # stop adding bands when the next saves less than this share of padded tokens


def state_prompt_lengths(record: TrajectoryRecord, tokenize: Callable[[str], int]) -> list[int]:
    """Token lengths of one record's six per-state prompts, in state order."""
    return [tokenize(render_state_prompt(record, state)) for state in VIOLATION_STATES]


def measure_state_lengths(records: list[TrajectoryRecord], tokenize: Callable[[str], int]) -> list[list[int]]:
    return [state_prompt_lengths(record, tokenize) for record in records]


def prompt_token_length(record: TrajectoryRecord, tokenize: Callable[[str], int]) -> int:
    """One record's band-relevant length under record-level assignment: the
    longest of its six prompts, because shared band membership must fit the
    longest call."""
    return max(state_prompt_lengths(record, tokenize))


def measure_lengths(records: list[TrajectoryRecord], tokenize: Callable[[str], int]) -> list[int]:
    return [prompt_token_length(record, tokenize) for record in records]


def _ceil_to_multiple(n: int, multiple: int = SHAPE_MULTIPLE) -> int:
    return max(multiple, ((n + multiple - 1) // multiple) * multiple)


def propose_boundaries(lengths: list[int], k: int) -> list[int]:
    """k band ceilings from equal-count splits of the observed distribution,
    each rounded up to a shape-friendly multiple. The top ceiling always
    covers the observed maximum."""
    if not lengths or k < 1:
        raise ValueError("need at least one length and one band")
    ordered = sorted(lengths)
    n = len(ordered)
    ceilings = []
    for band in range(1, k + 1):
        end = max(0, min(n - 1, (band * n) // k - 1))
        ceilings.append(_ceil_to_multiple(ordered[end]))
    # Rounding to shape multiples can collapse adjacent ceilings; dedupe.
    return sorted(set(ceilings))


def assign_band(length: int, ceilings: list[int]) -> int:
    """Smallest ceiling that fits, which is what the runner pads to."""
    for ceiling in ceilings:
        if length <= ceiling:
            return ceiling
    raise ValueError(f"length {length} exceeds the top band ceiling {ceilings[-1]}")


def padding_waste(lengths: list[int], ceilings: list[int]) -> float:
    """Share of padded tokens that are padding: 1 - real/padded."""
    real = sum(lengths)
    padded = sum(assign_band(n, ceilings) for n in lengths)
    return 1.0 - (real / padded) if padded else 0.0


def banding_report(state_lengths: list[list[int]]) -> dict:
    """Candidate bandings with waste under BOTH assignment strategies, plus
    the deterministic proposal.

    Record-level assignment gives all six of a record's calls the record's
    band (simple bookkeeping, every call pads to the record's longest).
    Prompt-level assignment gives each rendered prompt its own smallest
    fitting ceiling, using the SAME compiled shapes, so it can only cost
    less padding. The six calls are independent, so nothing requires shared
    membership; the comparison measures what the simpler bookkeeping costs.
    Boundaries and the proposal rule key on record-level waste; if
    prompt-level wins materially on the freezable corpus, the assignment
    switches before the freeze.
    """
    record_maxes = [max(row) for row in state_lengths]
    all_prompts = [n for row in state_lengths for n in row]
    real_tokens = sum(all_prompts)
    candidates = []
    for k in BAND_COUNTS:
        ceilings = propose_boundaries(record_maxes, k)
        record_level_padded = sum(assign_band(max(row), ceilings) * len(row) for row in state_lengths)
        prompt_level_padded = sum(assign_band(n, ceilings) for n in all_prompts)
        candidates.append(
            {
                "bands_requested": k,
                "ceilings": ceilings,
                "padding_waste": round(1.0 - real_tokens / record_level_padded, 4),
                "padding_waste_prompt_level": round(1.0 - real_tokens / prompt_level_padded, 4),
                "padded_tokens": record_level_padded,
                "padded_tokens_prompt_level": prompt_level_padded,
            }
        )
    proposal = candidates[0]
    for previous, candidate in zip(candidates, candidates[1:]):
        saved = previous["padding_waste"] - candidate["padding_waste"]
        if saved < MARGINAL_WASTE_CUTOFF:
            break
        proposal = candidate
    ordered = sorted(record_maxes)
    return {
        "distribution": {
            "records": len(record_maxes),
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[(len(ordered) * 9) // 10],
            "max": ordered[-1],
        },
        "candidates": candidates,
        "proposal": proposal,
        "proposal_rule": (
            f"smallest band count where the next band saves less than {MARGINAL_WASTE_CUTOFF:.0%} of "
            "padded tokens (record-level assignment); each band is one XLA shape at 20-30 min cold compile. "
            "The band count is derived by this rule, never chosen; the rule was fixed before the freezable "
            "data existed."
        ),
    }


def _hf_tokenize(name: str) -> Callable[[str], int]:
    """Real tokenizer path. Imported lazily so everything above runs and
    tests without transformers installed."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Install the dev extra (uv pip install -e '.[dev]') "
            "or pass lengths through a different tokenizer."
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        raise SystemExit(
            f"Could not load tokenizer {name!r}: {exc}\n"
            "Gemma checkpoints are gated on Hugging Face; run huggingface-cli login or set HF_TOKEN. "
            "This friction belongs in the E4 friction log."
        ) from exc
    return lambda text: len(tokenizer.encode(text))


def main(args: argparse.Namespace) -> None:
    records = load_staged(args.corpus) if args.staged else load_corpus(args.corpus)
    if not records:
        console.print(f"[yellow]No records loaded from {args.corpus}; nothing to measure.[/yellow]")
        return
    tokenize = _hf_tokenize(args.tokenizer)
    state_lengths = measure_state_lengths(records, tokenize)
    report = banding_report(state_lengths)
    # Per-record lengths ride in the report so the TPU runner bands by
    # lookup against the frozen measurement instead of re-tokenizing, which
    # keeps band membership identical between planning and execution.
    report["record_lengths"] = {record.id: max(row) for record, row in zip(records, state_lengths)}
    report["metadata"] = {
        "corpus": str(args.corpus),
        "staged": bool(args.staged),
        "tokenizer": args.tokenizer,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
    }

    d = report["distribution"]
    console.print(
        f"Prompt token lengths over {d['records']} records: min {d['min']}, p50 {d['p50']}, "
        f"p90 {d['p90']}, max {d['max']} (tokenizer: {args.tokenizer})"
    )
    table = Table(title="Candidate bandings")
    table.add_column("Bands", justify="right")
    table.add_column("Ceilings")
    table.add_column("Waste (record-level)", justify="right")
    table.add_column("Waste (prompt-level)", justify="right")
    for candidate in report["candidates"]:
        marker = " <- proposed" if candidate is report["proposal"] else ""
        table.add_row(
            str(len(candidate["ceilings"])),
            str(candidate["ceilings"]),
            f"{candidate['padding_waste']:.1%}{marker}",
            f"{candidate['padding_waste_prompt_level']:.1%}",
        )
    console.print(table)
    console.print(f"[dim]{report['proposal_rule']}[/dim]")
    if args.staged:
        console.print(
            "[yellow]Staged input: these boundaries are a preview. Freezable boundaries come from the "
            "adjudicated corpus with the target tokenizer.[/yellow]"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    console.print(f"[green]Band report written to {out_path}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token-length distribution and band proposal")
    parser.add_argument("--corpus", default="sample_data/trajectories.jsonl", help="Records to measure")
    parser.add_argument("--staged", action="store_true", help="Input is a staged (unadjudicated) candidate file")
    parser.add_argument("--tokenizer", default="google/gemma-4-E4B-it", help="Hugging Face tokenizer id")
    parser.add_argument("--output", default="results/band_report.json", help="Where the JSON report lands")
    main(parser.parse_args())
