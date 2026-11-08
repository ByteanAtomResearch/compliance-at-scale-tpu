"""
Module 6: Parity scorer. The same judge, two backends, one comparison.

parity.py reads two verdict files written by 05's runner or 06's run.py and
reports, state by state, how often the two runs agreed. It refuses to
compare two runs that differ in anything other than the backend: model,
prompt version and hash, structured-output schema hashes, record schema
fingerprint, call mode, sampling parameters, selected band ceilings and
max_model_len must match, and the band report hash and dtype must match
when both files carry them. A parity number across two different prompts
would look like a backend difference and be nothing of the kind. The two
files must also cover the same record ids, because 05's scorer treats a
missing id as a negative without saying so.

What it reports. Per state: n decisions (one per record), the agreement
count and rate with a Wilson interval, the 2x2 split (both detected, left
only, right only, neither), Cohen's kappa, and flags for small samples and
sparse positives, where kappa is unstable and the counts are the result.
Agreement is on raw detections, what the judge said; malformation is
reported separately per side so the two questions stay apart. Per side:
parse errors, the sentinel (-1) count, the malformation report when a
corpus is given, provenance, and the per-band wall times as information.
With --labels, precision and recall per tier and state on each side and
the delta between them, plus authorization_bypass recall by variant.

Bit-identical verdicts are not expected across backends, which is why
kappa is the headline and exact match is not. Running parity on two runs
of the same backend is the determinism control for that headline.

Usage:
    python 06_misreported_state_torchtpu/parity.py \
        --left results/06_verdicts_jax.json --left-name jax \
        --right results/06_verdicts_torchtpu.json --right-name torchtpu \
        --labels results/06_eval_corpus.jsonl --output results/06_parity.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import lab05
from rich.console import Console
from rich.table import Table

_score = lab05.load("score")
_corpus = lab05.load("corpus")
_schema = lab05.load("schema")

console = Console()

# Field names come from 05's build_run_metadata. A test cross-checks every
# name here against what that function emits, so a typo cannot turn the
# guard into a no-op. max_model_len is the one addition beyond the spec: it
# can diverge through the MAX_MODEL_LEN override while the ceilings agree.
REQUIRED_MATCH: tuple[str, ...] = (
    "model",
    "prompt_version",
    "prompt_hash",
    "structured_outputs_schema_hash",
    "record_schema_fingerprint",
    "call_mode",
    "sampling_params",
    "band_ceilings_selected",
    "max_model_len",
)
# 06's run.py always writes these; 05's runner never does. A dtype difference
# between backends is a confound, not a backend property, so it refuses too.
MATCH_WHEN_BOTH_PRESENT: tuple[str, ...] = ("band_report_sha256", "dtype")
INFORMATIONAL: tuple[str, ...] = (
    "backend",
    "git",
    "model_revision",
    "container_image_digest",
    "vllm_distribution",
    "tensor_parallel_size",
    "mode",
    "evaluator",
    "runner",
    "xla_cache_path_env",
)
SMALL_SAMPLE_N = 10
SPARSE_POSITIVES = 5
KAPPA_UNDEFINED_NOTE = "undefined: both runs unanimous for this state, so expected agreement is 1; read the counts"
SMALL_SAMPLE_NOTE = f"fewer than {SMALL_SAMPLE_N} decisions: read the counts, not kappa"
SPARSE_NOTE = f"a side has fewer than {SPARSE_POSITIVES} positives or negatives: kappa swings on one record"
NOT_COMPARED_ONE_SIDE = "recorded on one side only; not compared"
NOT_COMPARED_ABSENT = "absent on both sides; not compared"

_MISSING = object()


# ── Loading and gates ─────────────────────────────────────────────────────────


def load_run(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if "verdicts" not in doc:
        raise SystemExit(f"{path}: no 'verdicts' key; parity needs an --eval-set run")
    return doc


def _show(value: Any) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def metadata_mismatches(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Every field that differs, never just the first. A missing field counts as a mismatch."""
    mismatches: list[str] = []
    for key in REQUIRED_MATCH:
        left_value, right_value = left.get(key, _MISSING), right.get(key, _MISSING)
        if left_value is _MISSING or right_value is _MISSING or left_value != right_value:
            mismatches.append(f"{key}: left={_show(left_value)} right={_show(right_value)}")
    for key in MATCH_WHEN_BOTH_PRESENT:
        if key in left and key in right and left[key] != right[key]:
            mismatches.append(f"{key}: left={left[key]!r} right={right[key]!r}")
    return mismatches


def id_mismatches(left_verdicts: dict[str, Any], right_verdicts: dict[str, Any]) -> list[str]:
    """score.evaluate treats a missing id as a negative silently; parity refuses instead."""
    left_only = sorted(set(left_verdicts) - set(right_verdicts))
    right_only = sorted(set(right_verdicts) - set(left_verdicts))
    mismatches: list[str] = []
    if left_only:
        mismatches.append(f"{len(left_only)} record ids only in the left file: {left_only[:20]}")
    if right_only:
        mismatches.append(f"{len(right_only)} record ids only in the right file: {right_only[:20]}")
    return mismatches


def compared_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """The configuration both runs share, after the gate passed."""
    out: dict[str, Any] = {key: left.get(key) for key in REQUIRED_MATCH}
    for key in MATCH_WHEN_BOTH_PRESENT:
        if key in left and key in right:
            out[key] = left[key]
        elif key in left or key in right:
            out[key] = NOT_COMPARED_ONE_SIDE
        else:
            out[key] = NOT_COMPARED_ABSENT
    return out


# ── Agreement ─────────────────────────────────────────────────────────────────


def cohens_kappa(both: int, left_only: int, right_only: int, neither: int) -> float | None:
    """Cohen's kappa from the 2x2 split, in integers until the last division.

    Undefined (None) when expected agreement is 1, which happens only when
    both runs are unanimous in the same direction. One-sided unanimity is
    defined and comes out as 0.0.
    """
    n = both + left_only + right_only + neither
    if n == 0:
        return None
    expected = (both + left_only) * (both + right_only) + (right_only + neither) * (left_only + neither)
    numerator = n * (both + neither) - expected
    denominator = n * n - expected
    return None if denominator == 0 else numerator / denominator


def state_agreement(
    left_verdicts: dict[str, Any], right_verdicts: dict[str, Any], ids: list[str], state: str
) -> dict[str, Any]:
    both = left_only = right_only = neither = 0
    disagree: list[str] = []
    for record_id in ids:
        left_hit = state in _score.detected_states(left_verdicts[record_id])
        right_hit = state in _score.detected_states(right_verdicts[record_id])
        if left_hit and right_hit:
            both += 1
        elif left_hit:
            left_only += 1
            disagree.append(record_id)
        elif right_hit:
            right_only += 1
            disagree.append(record_id)
        else:
            neither += 1
    n = len(ids)
    agree = both + neither
    left_positives, right_positives = both + left_only, both + right_only
    kappa = cohens_kappa(both, left_only, right_only, neither)
    sparse = n > 0 and min(left_positives, n - left_positives, right_positives, n - right_positives) < SPARSE_POSITIVES
    notes = []
    if kappa is None:
        notes.append(KAPPA_UNDEFINED_NOTE)
    if n < SMALL_SAMPLE_N:
        notes.append(SMALL_SAMPLE_NOTE)
    if sparse:
        notes.append(SPARSE_NOTE)
    return {
        "n": n,
        "agree": agree,
        "agreement_rate": (agree / n) if n else None,
        "agreement_wilson": list(_score.wilson_interval(agree, n)),
        "both_detected": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "left_positives": left_positives,
        "right_positives": right_positives,
        "disagree_ids": sorted(disagree),
        "kappa": kappa,
        "kappa_note": KAPPA_UNDEFINED_NOTE if kappa is None else None,
        "small_sample": n < SMALL_SAMPLE_N,
        "sparse_positives": sparse,
        "notes": notes,
    }


def agreement_by_state(left_verdicts: dict[str, Any], right_verdicts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ids = sorted(set(left_verdicts) & set(right_verdicts))
    return {state: state_agreement(left_verdicts, right_verdicts, ids, state) for state in _schema.VIOLATION_STATES}


def overall_agreement(per_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = sum(entry["n"] for entry in per_state.values())
    agree = sum(entry["agree"] for entry in per_state.values())
    return {
        "decisions": n,
        "agree": agree,
        "agreement_rate": (agree / n) if n else None,
        "agreement_wilson": list(_score.wilson_interval(agree, n)),
    }


# ── Per-side summaries ────────────────────────────────────────────────────────


def side_summary(doc: dict[str, Any], records: list | None) -> dict[str, Any]:
    """What one run says about itself. Nothing here is compared or gated."""
    metadata = doc.get("metadata", {})
    verdicts = doc.get("verdicts", {})
    return {
        "parse_errors": metadata.get("parse_errors"),
        "sentinel_unlocalized": _score.sentinel_audit(verdicts),
        "malformation": _score.malformation_report(verdicts, records) if records is not None else None,
        "provenance_recorded": _score.provenance_recorded(metadata),
        "informational": {key: metadata.get(key) for key in INFORMATIONAL},
        "bands": [
            {
                "ceiling": band.get("ceiling"),
                "records": band.get("records"),
                "prompts": band.get("prompts"),
                "wall_times_seconds": band.get("wall_times_seconds"),
            }
            for band in doc.get("bands", [])
        ],
    }


# ── Labelled comparison ───────────────────────────────────────────────────────


def load_label_records(path: str | Path, allow_staged: bool, verdict_ids: set[str]) -> tuple[list, list[str]]:
    """The adjudicated corpus, cut down to the records both runs judged.

    --allow-staged mirrors score.py: proposed labels stand in for decisions,
    for synthetic verification only. Any verdict id without a record is a
    refusal, and records without a verdict are excluded and listed, so
    score.evaluate never sees a silent negative.
    """
    try:
        records = _corpus.load_records(path, require_adjudicated=not allow_staged)
    except _corpus.CorpusIntegrityError as exc:
        raise SystemExit(f"{exc}\nPass --allow-staged to score proposed labels (synthetic verification only).") from exc
    if not records:
        raise SystemExit(f"no records loaded from {path}")
    if allow_staged:
        for record in records:
            record.labels = record.labels or record.proposed_labels
    by_id = {record.id: record for record in records}
    missing = sorted(verdict_ids - set(by_id))
    if missing:
        raise SystemExit(f"{len(missing)} verdict ids have no record in {path}: {missing[:20]}")
    kept = [record for record in records if record.id in verdict_ids]
    excluded = sorted(set(by_id) - verdict_ids)
    return kept, excluded


_HEADLINE_FIELDS = (
    "tp",
    "fp",
    "fn",
    "tn",
    "positives",
    "negatives",
    "recall",
    "recall_wilson",
    "precision",
    "precision_wilson",
)


def _headline(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {field: evaluation["headline"].get(field) for field in _HEADLINE_FIELDS}


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else right - left


def labelled_comparison(records: list, left_verdicts: dict[str, Any], right_verdicts: dict[str, Any]) -> dict[str, Any]:
    """score.evaluate per tier and side, then right minus left. Tiers are split first, as score.main does."""
    tiers: dict[str, dict[str, Any]] = {}
    for tier in _schema.TIERS:
        tier_records = [record for record in records if record.tier == tier]
        if not tier_records:
            continue
        left_eval = _score.evaluate(tier_records, left_verdicts, "judge")
        right_eval = _score.evaluate(tier_records, right_verdicts, "judge")
        per_state: dict[str, Any] = {}
        for state in _schema.VIOLATION_STATES:
            left, right = _headline(left_eval[state]), _headline(right_eval[state])
            per_state[state] = {
                "left": left,
                "right": right,
                "delta": {metric: _delta(left[metric], right[metric]) for metric in ("recall", "precision")},
                "malformed_pairs": {
                    "left": left_eval[state]["malformed_pairs"],
                    "right": right_eval[state]["malformed_pairs"],
                },
                "reduced_region": {
                    "left": left_eval[state]["reduced_region"],
                    "right": right_eval[state]["reduced_region"],
                },
            }
        tiers[tier] = per_state
    return {
        "tiers": tiers,
        "variant_recall": {
            "left": _score.variant_recall(records, left_verdicts),
            "right": _score.variant_recall(records, right_verdicts),
        },
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt_kappa(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_report(result: dict[str, Any]) -> None:
    left_name, right_name = result["left"]["name"], result["right"]["name"]
    console.print(
        f"[bold]Configuration identical[/bold] on {len(REQUIRED_MATCH)} guarded fields; "
        f"{result['records']['n']} shared records, {result['overall']['decisions']} state decisions."
    )
    table = Table(title=f"Per-state agreement: {left_name} vs {right_name}")
    for column in ("State", "n", "Agree", "Rate", "Wilson", "Kappa", f"{left_name}+", f"{right_name}+", "Notes"):
        table.add_column(column)
    for state, entry in result["agreement"].items():
        table.add_row(
            state,
            str(entry["n"]),
            str(entry["agree"]),
            _score._fmt(entry["agreement_rate"]),
            _score._fmt_wilson(tuple(entry["agreement_wilson"])),
            _fmt_kappa(entry["kappa"]),
            str(entry["left_positives"]),
            str(entry["right_positives"]),
            "; ".join(entry["notes"]),
        )
    console.print(table)
    for side in ("left", "right"):
        summary = result["per_side"][side]
        malformation = summary["malformation"]
        rate = "-" if malformation is None else f"{malformation['rate']:.1%}"
        console.print(
            f"  {result[side]['name']}: parse_errors={summary['parse_errors']}, "
            f"sentinel_unlocalized={summary['sentinel_unlocalized']}, malformation={rate}, "
            f"provenance_recorded={summary['provenance_recorded']}"
        )
    labels = result["labels"]
    if labels:
        for tier, per_state in labels["tiers"].items():
            table = Table(title=f"Judge vs labels, tier {tier} ({labels['source']})")
            for column in (
                "State",
                f"recall {left_name}",
                f"recall {right_name}",
                "delta",
                f"precision {left_name}",
                f"precision {right_name}",
                "delta",
            ):
                table.add_column(column)
            for state, entry in per_state.items():
                table.add_row(
                    state,
                    _score._fmt(entry["left"]["recall"]),
                    _score._fmt(entry["right"]["recall"]),
                    _score._fmt(entry["delta"]["recall"]),
                    _score._fmt(entry["left"]["precision"]),
                    _score._fmt(entry["right"]["precision"]),
                    _score._fmt(entry["delta"]["precision"]),
                )
            console.print(table)
        excluded = len(labels["excluded_unverdicted_ids"])
        if excluded:
            console.print(f"  [yellow]{excluded} labelled records had no verdict and were excluded[/yellow]")


# ── Entry ─────────────────────────────────────────────────────────────────────


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Gate first, then measure. Raises SystemExit before anything is computed when the runs differ."""
    left_doc, right_doc = load_run(args.left), load_run(args.right)
    left_meta, right_meta = left_doc.get("metadata", {}), right_doc.get("metadata", {})
    left_verdicts, right_verdicts = left_doc["verdicts"], right_doc["verdicts"]
    left_name = args.left_name or left_meta.get("backend") or "left"
    right_name = args.right_name or right_meta.get("backend") or "right"

    mismatches = metadata_mismatches(left_meta, right_meta) + id_mismatches(left_verdicts, right_verdicts)
    if mismatches:
        raise SystemExit(
            "refusing to compare: the two runs differ in more than the backend\n  " + "\n  ".join(mismatches)
        )

    records: list | None = None
    labels: dict[str, Any] | None = None
    if args.labels:
        records, excluded = load_label_records(args.labels, args.allow_staged, set(left_verdicts))
        labels = {
            "path": str(args.labels),
            "source": "proposed labels (--allow-staged; synthetic verification only)"
            if args.allow_staged
            else "adjudicated",
            "excluded_unverdicted_ids": excluded,
            **labelled_comparison(records, left_verdicts, right_verdicts),
        }

    per_state = agreement_by_state(left_verdicts, right_verdicts)
    return {
        "left": {"name": left_name, "path": str(args.left)},
        "right": {"name": right_name, "path": str(args.right)},
        "compared_metadata": compared_metadata(left_meta, right_meta),
        "records": {"n": len(left_verdicts), "ids_identical": True},
        "per_side": {"left": side_summary(left_doc, records), "right": side_summary(right_doc, records)},
        "agreement": per_state,
        "overall": overall_agreement(per_state),
        "labels": labels,
        "note": (
            "configuration identity was verified before any number was computed; backend, git sha, "
            "distribution, provenance and wall times are reported, never compared"
        ),
    }


def main(args: argparse.Namespace) -> None:
    result = compare(args)
    print_report(result)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        console.print(f"[green]Parity report written to {out_path}[/green]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 6: compare two judge runs that differ only in backend")
    parser.add_argument("--left", required=True, help="Verdict file from one backend")
    parser.add_argument("--right", required=True, help="Verdict file from the other backend")
    parser.add_argument("--left-name", default=None, help="Label for the left run; defaults to its recorded backend")
    parser.add_argument("--right-name", default=None, help="Label for the right run; defaults to its recorded backend")
    parser.add_argument("--labels", default=None, help="Adjudicated corpus for precision and recall per side")
    parser.add_argument(
        "--allow-staged", action="store_true", help="Score proposed labels (synthetic verification only)"
    )
    parser.add_argument("--output", default=None, help="Where to write the parity JSON; nothing is written on refusal")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
