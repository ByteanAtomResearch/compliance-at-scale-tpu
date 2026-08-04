"""
Module 5: Batch trajectory evaluation on Cloud TPU with vLLM.

Mirrors 02_offline_batch/batch_rai_eval.py: build prompts, one
llm.generate() per band with per-prompt structured outputs, tolerant
parsing, a metadata/summary report. What Part 2 adds is band discipline
(one XLA shape per band, membership frozen in the band report) and run
metadata complete enough that every number in a published table traces to
the exact schema, prompt, model, and cache state that produced it.

Modes:
    --dry-run       build the full execution plan locally, no vllm import,
                    no TPU. This is how the harness is verified off-TPU.
    --compile-only  warm the XLA cache for the selected band(s) with a
                    2-record batch; explicitly records that no measurement
                    was taken.
    (default)       timed benchmark: --repeat N generate() passes per band,
                    wall times recorded raw, medians left to reporting.
    --eval-set      judge evaluation over the eval records; writes verdicts
                    JSON shaped exactly like rules_baseline.py output so
                    score.py consumes both identically.

Sequence-length versus batch-size effects (Gate 4 Item 2): raising
max_model_len to clear a band enlarges KV-cache allocation per sequence,
which shrinks the batch that fits in HBM, which lowers throughput
independently of sequence length. Every run therefore records
max_model_len, submitted batch size, and whatever runtime batch/memory
stats the vLLM build exposes, so the long band's slowdown can be
attributed rather than blended.

IMPORTANT: first run per band compiles for 20-30 minutes on v5e-4. Cache
lands in ~/.cache/vllm/xla_cache; push it to GCS before teardown
(cache_helper.sh).

Usage:
    python 05_trajectory_eval/batch_trajectory_eval.py --dry-run --staged \
        --input sample_data/staging/candidates_tier2_batch1.jsonl \
        --bands-report results/band_report.json
    make trajectory ARGS="--band 1728 --limit 2 --compile-only"
    make trajectory ARGS="--band all --repeat 3"
    make trajectory ARGS="--eval-set --repeat 1"
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from corpus import load_corpus, load_staged
from prompts import PROMPT_VERSION, prompt_hash, render_multi_prompt, render_state_prompt
from rich.console import Console
from schema import (
    STATE_VERDICT_SCHEMA,
    VERDICT_SCHEMA,
    VIOLATION_STATES,
    Verdict,
    Violation,
    derive_enforcement,
    record_schema_fingerprint,
    schema_hash,
    validate_verdict,
    verdict_to_dict,
)

console = Console()

# Verdict generations are structured and short; this cap covers evidence
# strings without letting a runaway response distort a band's shape.
VERDICT_MAX_TOKENS = 384

# Enforcement is the shared operator-policy lookup (schema.derive_enforcement),
# applied after the model's violations are assembled. No prompt or model-facing
# schema mentions enforcement.


# ── Band plumbing ─────────────────────────────────────────────────────────────


def load_band_report(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if "record_lengths" not in report:
        raise SystemExit(f"{path} has no record_lengths; re-run bands.py to regenerate it.")
    return report


def selected_ceilings(report: dict, band_arg: str) -> list[int]:
    """--band takes a ceiling number or 'all'. Names are the ceilings
    themselves so the flag, the report, and the results all say the same
    thing."""
    proposal = report["proposal"]["ceilings"]
    if band_arg == "all":
        return list(proposal)
    try:
        ceiling = int(band_arg)
    except ValueError as exc:
        raise SystemExit(f"--band must be 'all' or one of {proposal}, got {band_arg!r}") from exc
    if ceiling not in proposal:
        raise SystemExit(f"--band {ceiling} is not a proposed ceiling; proposed: {proposal}")
    return [ceiling]


def records_for_ceiling(records: list, report: dict, ceiling: int) -> list:
    """Band membership by lookup against the frozen measurement."""
    ceilings = report["proposal"]["ceilings"]
    lengths = report["record_lengths"]
    members = []
    for record in records:
        if record.id not in lengths:
            raise SystemExit(f"record {record.id} missing from the band report; re-run bands.py on this corpus.")
        length = lengths[record.id]
        fit = next((c for c in ceilings if length <= c), None)
        if fit == ceiling:
            members.append(record)
    return members


def compute_max_model_len(top_ceiling: int) -> int:
    """Prompt ceiling plus verdict headroom, shape-rounded. Env override for
    experiments; the effective value is always recorded in metadata."""
    needed = top_ceiling + VERDICT_MAX_TOKENS
    rounded = ((needed + 63) // 64) * 64
    return int(os.environ.get("MAX_MODEL_LEN", str(rounded)))


# ── Prompt and verdict assembly ───────────────────────────────────────────────


def build_prompts(records: list, single_call: bool) -> list[tuple[str, str, str]]:
    """(record_id, state_or_mode, prompt) triples. Six-call is primary; the
    single-call path exists for the anchoring comparison."""
    triples = []
    for record in records:
        if single_call:
            triples.append((record.id, "multi", render_multi_prompt(record)))
        else:
            for state in VIOLATION_STATES:
                triples.append((record.id, state, render_state_prompt(record, state)))
    return triples


def parse_response(raw_text: str) -> dict[str, Any]:
    """Same tolerant parse as Module 2: structured outputs should make this
    a no-op, and the fallback catches fence-wrapped or truncated responses."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_response": raw_text}


def assemble_six_call_verdict(state_responses: dict[str, dict[str, Any]]) -> Verdict:
    """Six per-state responses into one trajectory verdict. Responses with
    parse errors contribute nothing (and are counted upstream); detections
    keep the judge's own step index and confidence."""
    violations = []
    for state in VIOLATION_STATES:
        response = state_responses.get(state, {})
        if response.get("parse_error") or not response.get("detected"):
            continue
        violations.append(
            Violation(
                state=state,
                failed_step_index=int(response.get("failed_step_index", -1)),
                evidence=str(response.get("evidence", "")),
                confidence=float(response.get("confidence", 0.0)),
            )
        )
    verdict = Verdict(violations=violations, recommended_enforcement=derive_enforcement(violations))
    validate_verdict(verdict)
    return verdict


def assemble_single_call_verdict(response: dict[str, Any]) -> Verdict:
    if response.get("parse_error"):
        return Verdict(violations=[], recommended_enforcement="none")
    violations = [
        Violation(
            state=v["state"],
            failed_step_index=int(v.get("failed_step_index", -1)),
            evidence=str(v.get("evidence", "")),
            confidence=float(v.get("confidence", 0.0)),
        )
        for v in response.get("violations", [])
        if v.get("state") in VIOLATION_STATES
    ]
    verdict = Verdict(violations=violations, recommended_enforcement=derive_enforcement(violations))
    validate_verdict(verdict)
    return verdict


# ── Run metadata ──────────────────────────────────────────────────────────────


def xla_cache_state() -> dict[str, Any]:
    cache_dir = Path.home() / ".cache" / "vllm" / "xla_cache"
    populated = cache_dir.exists() and any(cache_dir.iterdir())
    return {
        "xla_cache_dir": str(cache_dir),
        "xla_cache_populated_before_run": bool(populated),
        "note": "populated cache suggests warm start; band shape changes still recompile",
    }


def build_run_metadata(args: argparse.Namespace, ceilings: list[int], max_model_len: int, mode: str) -> dict:
    return {
        "mode": mode,
        "model": args.model,
        "model_revision": os.environ.get("MODEL_REVISION", "unpinned (record before publishing)"),
        "container_image_digest": os.environ.get("CONTAINER_IMAGE_DIGEST", "unrecorded (set before publishing)"),
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": VERDICT_MAX_TOKENS},
        "structured_outputs_schema_hash": {
            "six_call": schema_hash(STATE_VERDICT_SCHEMA),
            "single_call": schema_hash(VERDICT_SCHEMA),
        },
        "record_schema_fingerprint": record_schema_fingerprint(),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "call_mode": "single_call" if args.single_call else "six_call",
        "band_ceilings_selected": ceilings,
        "max_model_len": max_model_len,
        "cache": xla_cache_state(),
    }


# ── TPU execution (imports vllm; never reached under --dry-run) ───────────────


def run_band(records: list, args: argparse.Namespace, ceiling: int, max_model_len: int) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    triples = build_prompts(records, args.single_call)
    prompts = [t[2] for t in triples]
    schema = VERDICT_SCHEMA if args.single_call else STATE_VERDICT_SCHEMA
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=VERDICT_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=schema),
    )

    try:
        import jax

        chip_count = len(jax.devices("tpu"))
    except Exception:
        chip_count = int(os.environ.get("TPU_CHIPS", "4"))
    chip_count = int(os.environ.get("TPU_CHIPS", str(chip_count)))

    console.print(
        f"[bold]Band {ceiling}[/bold]: {len(records)} records, {len(prompts)} prompts, "
        f"max_model_len={max_model_len}. First run per shape compiles 20-30 min."
    )
    llm = LLM(model=args.model, tensor_parallel_size=chip_count, dtype="bfloat16", max_model_len=max_model_len)

    runtime_stats: dict[str, Any] = {"submitted_batch_size": len(prompts)}
    try:
        cache_config = llm.llm_engine.cache_config
        runtime_stats["num_gpu_blocks"] = getattr(cache_config, "num_gpu_blocks", None)
        runtime_stats["block_size"] = getattr(cache_config, "block_size", None)
    except Exception:
        runtime_stats["runtime_note"] = "this vllm build exposes no cache/batch internals; recorded submitted size only"

    repeats = 1 if args.compile_only else args.repeat
    wall_times = []
    outputs = None
    for i in range(repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        wall_times.append(round(time.perf_counter() - start, 3))
        console.print(f"  pass {i + 1}/{repeats}: {wall_times[-1]}s")

    result: dict[str, Any] = {
        "ceiling": ceiling,
        "records": len(records),
        "prompts": len(prompts),
        "runtime": runtime_stats,
    }
    if args.compile_only:
        result["compile_only"] = True
        result["measurement"] = "none taken; this run existed to warm the XLA cache"
    else:
        result["wall_times_seconds"] = wall_times
    if args.eval_set and outputs is not None:
        raw = [output.outputs[0].text for output in outputs]
        result["responses"] = {(t[0], t[1]): r for t, r in zip(triples, raw)}
    return result


def verdicts_from_band_results(band_results: list[dict], single_call: bool) -> tuple[dict, int]:
    """Collect per-record verdicts from eval-set band runs."""
    by_record: dict[str, dict[str, dict]] = {}
    parse_errors = 0
    for band in band_results:
        for (record_id, state), raw in band.get("responses", {}).items():
            parsed = parse_response(raw)
            if parsed.get("parse_error"):
                parse_errors += 1
            by_record.setdefault(record_id, {})[state] = parsed
    verdicts = {}
    for record_id, responses in by_record.items():
        if single_call:
            verdicts[record_id] = verdict_to_dict(assemble_single_call_verdict(responses.get("multi", {})))
        else:
            verdicts[record_id] = verdict_to_dict(assemble_six_call_verdict(responses))
    return verdicts, parse_errors


# ── Entry ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    records = load_staged(args.input) if args.staged else load_corpus(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        console.print(f"[yellow]No records loaded from {args.input}; nothing to run.[/yellow]")
        return
    report = load_band_report(args.bands_report)
    ceilings = selected_ceilings(report, args.band)
    max_model_len = compute_max_model_len(max(ceilings))
    mode = (
        "dry_run"
        if args.dry_run
        else ("compile_only" if args.compile_only else ("eval_set" if args.eval_set else "benchmark"))
    )
    metadata = build_run_metadata(args, ceilings, max_model_len, mode)

    plan = []
    for ceiling in ceilings:
        members = records_for_ceiling(records, report, ceiling)
        triples = build_prompts(members, args.single_call)
        plan.append({"ceiling": ceiling, "records": len(members), "prompts": len(triples)})

    if args.dry_run:
        output = {
            "metadata": metadata,
            "plan": plan,
            "executed": False,
            "note": "dry run: no vllm import, no TPU, no measurement",
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        console.print(
            f"[bold]Dry run.[/bold] mode that WOULD execute: "
            f"{'single-call' if args.single_call else 'six-call'}, bands {ceilings}, "
            f"max_model_len {max_model_len}."
        )
        for entry in plan:
            console.print(f"  band {entry['ceiling']}: {entry['records']} records -> {entry['prompts']} prompts")
        console.print(f"[green]Plan written to {out_path}. No TPU run occurred.[/green]")
        return

    band_results = []
    for ceiling in ceilings:
        members = records_for_ceiling(records, report, ceiling)
        if not members:
            console.print(f"[yellow]Band {ceiling}: no records, skipping.[/yellow]")
            continue
        band_results.append(run_band(members, args, ceiling, max_model_len))

    output = {
        "metadata": metadata,
        "bands": [{k: v for k, v in band.items() if k != "responses"} for band in band_results],
    }
    if args.eval_set:
        verdicts, parse_errors = verdicts_from_band_results(band_results, args.single_call)
        output["verdicts"] = verdicts
        output["metadata"]["parse_errors"] = parse_errors
        output["metadata"]["evaluator"] = "gemma_judge"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    console.print(f"[green]Results written to {out_path}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch trajectory evaluation on Cloud TPU")
    parser.add_argument("--model", default=os.environ.get("MODEL", "google/gemma-4-E4B-it"))
    parser.add_argument("--input", default="sample_data/trajectories.jsonl")
    parser.add_argument("--staged", action="store_true", help="Input is a staged (unadjudicated) file")
    parser.add_argument("--bands-report", default="results/band_report.json")
    parser.add_argument("--band", default="all", help="A proposed ceiling, or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--eval-set", action="store_true")
    parser.add_argument("--single-call", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="results/trajectory_run.json")
    parsed = parser.parse_args()
    # Eval-set runs land where make score expects them unless overridden.
    if parsed.eval_set and parsed.output == "results/trajectory_run.json":
        parsed.output = "results/trajectory_verdicts.json"
    main(parsed)
