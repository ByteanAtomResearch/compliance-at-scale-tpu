"""
Module 6: The trajectory judge on a declared vLLM backend (Part 3).

Both backends are reached the same way in Python, `from vllm import LLM`.
Which one runs is decided by the environment the command runs in: the
vllm-tpu container for JAX, the vllm-torchtpu venv for TorchTPU. No flag can
switch that. So --backend switches nothing. It records the backend in the
run metadata and refuses to start when the installed vllm distribution does
not match, because a flag that claimed to switch backends would be exactly
the misreported state this module measures.

Why this file carries its own generate loop. 05's run_band builds LLM(...)
with dtype="bfloat16" inside the function, so 06 cannot reuse it and still
choose a dtype. run_band below is a copy of 05's with dtype and the chip
count as parameters. Everything else (loading, banding, prompt building,
metadata, verdict assembly) is 05's code loaded by path through lab05.
tests/test_misreport_runner.py runs 05's real run_band and this one under
the same fakes and asserts that the sampling settings are identical, field
by field. If that test fails, parity is off.

Every run log also records the commit that ran (sha, branch, dirty flag), a
hash of the band report, and the vllm distribution found, so a number can be
traced back to the exact code and inputs that produced it. It records where
the judged records' labels came from too (labels.py). The judge never reads
a label, so that block changes no verdict; it is there so whoever scores the
run knows whose labels they are scoring against.

Usage mirrors 05's runner plus --backend and --dtype:
    python 06_misreported_state_torchtpu/run.py --backend torchtpu \
        --model RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic --staged \
        --input sample_data/staging/candidates_tier2_batch1.jsonl \
        --bands-report results/smoke_bands.json --limit 2 --dry-run \
        --output results/run06_plan.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import lab05
from rich.console import Console

_runner = lab05.load("batch_trajectory_eval")
_corpus = lab05.load("corpus")
_schema = lab05.load("schema")
_labels = lab05.sibling("labels")

console = Console()

# The import name is `vllm` for both; the distribution name tells them apart.
BACKEND_DISTRIBUTIONS: dict[str, str] = {"jax": "vllm-tpu", "torchtpu": "vllm-torchtpu"}
CUDA_DISTRIBUTION = "vllm"
DTYPES: tuple[str, ...] = ("bfloat16", "auto")
DEFAULT_MODEL = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
GIT_UNAVAILABLE = "unavailable (not a git checkout)"
RUNNER_PATH = Path(__file__).resolve().relative_to(lab05.REPO_ROOT).as_posix()

# The one source of truth for generation settings. run_band splats this into
# SamplingParams and extra_metadata copies it into the run log, so the two
# cannot drift. 05 writes the same three values by hand in two places.
SAMPLING: dict[str, float | int] = {"temperature": 0.0, "top_p": 1.0, "max_tokens": _runner.VERDICT_MAX_TOKENS}


# ── Backend guard ─────────────────────────────────────────────────────────────


def installed_vllm_distributions() -> list[tuple[str, str]]:
    """Which vllm distributions this environment holds, by metadata only.

    Imports no vllm: a dry run on a laptop must never pull in an accelerator
    stack, and the guard has to run before the model is even named.
    """
    found: list[tuple[str, str]] = []
    for dist in (*BACKEND_DISTRIBUTIONS.values(), CUDA_DISTRIBUTION):
        try:
            found.append((dist, importlib.metadata.version(dist)))
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def require_backend(backend: str, installed: list[tuple[str, str]]) -> dict[str, str]:
    """Refuse to start unless exactly the requested backend's distribution is installed.

    Pure: takes the distribution list so tests can exercise every branch
    without touching importlib.metadata.
    """
    expected = BACKEND_DISTRIBUTIONS[backend]
    tpu = [(name, version) for name, version in installed if name in BACKEND_DISTRIBUTIONS.values()]
    if not tpu:
        hint = ""
        if any(name == CUDA_DISTRIBUTION for name, _ in installed):
            hint = f" The plain '{CUDA_DISTRIBUTION}' distribution is installed; this lab needs a TPU distribution."
        raise SystemExit(
            f"--backend {backend} needs the {expected} distribution, and neither "
            f"{' nor '.join(BACKEND_DISTRIBUTIONS.values())} is installed.{hint}"
        )
    if len(tpu) > 1:
        names = ", ".join(f"{name} {version}" for name, version in tpu)
        raise SystemExit(
            f"both TPU distributions are installed ({names}); the import name 'vllm' is ambiguous. "
            "One environment per backend."
        )
    name, version = tpu[0]
    if name != expected:
        other = next(b for b, dist in BACKEND_DISTRIBUTIONS.items() if dist == name)
        raise SystemExit(
            f"--backend {backend} expects {expected}, but {name} {version} is installed. "
            f"Pass --backend {other}, or run from the {expected} environment."
        )
    return {"name": name, "version": version}


# ── Run provenance ────────────────────────────────────────────────────────────


def resolve_tensor_parallel_size(probe_jax: bool) -> int:
    """05's chip-count rule, hoisted out of the generate loop so metadata can record it.

    Same outcome as 05's run_band: TPU_CHIPS wins whenever it is set;
    otherwise the jax device count, and 4 when jax is unavailable. Dry runs
    skip the jax probe so a laptop imports no accelerator stack, which means
    a dry run's value is the environment variable, not a measurement.
    """
    chip_count = int(os.environ.get("TPU_CHIPS", "4"))
    if probe_jax:
        try:
            import jax

            chip_count = len(jax.devices("tpu"))
        except Exception:
            chip_count = int(os.environ.get("TPU_CHIPS", "4"))
    return int(os.environ.get("TPU_CHIPS", str(chip_count)))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_provenance(repo_root: Path) -> dict[str, Any]:
    """Commit, branch and dirty flag of the checkout that ran, for the run log.

    GIT_SHA wins when set, because a VM running from a copied tree has no
    .git. A dirty tree gets a -dirty suffix so a run made during iteration
    cannot pass for a reproducible one. Never raises: a missing git binary or
    a directory that is not a checkout records that fact instead.
    """
    override = os.environ.get("GIT_SHA")

    def _git(*argv: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *argv], capture_output=True, text=True, check=True, timeout=10
        )
        return completed.stdout.strip()

    try:
        sha = override or _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"sha": override or GIT_UNAVAILABLE, "branch": GIT_UNAVAILABLE, "dirty": None}
    return {"sha": f"{sha}-dirty" if dirty else sha, "branch": branch, "dirty": dirty}


def extra_metadata(
    args: argparse.Namespace,
    tensor_parallel_size: int,
    vllm_distribution: dict[str, str],
    band_report_sha256: str,
    label_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What 06 adds to 05's build_run_metadata. sampling_params is replaced with SAMPLING itself."""
    return {
        "label_sources": label_sources,
        "backend": args.backend,
        "dtype": args.dtype,
        "checkpoint": args.model,
        "tensor_parallel_size": tensor_parallel_size,
        "vllm_distribution": vllm_distribution,
        "band_report_sha256": band_report_sha256,
        "xla_cache_path_env": os.environ.get("VLLM_XLA_CACHE_PATH"),
        "git": git_provenance(lab05.REPO_ROOT),
        "runner": RUNNER_PATH,
        "sampling_params": dict(SAMPLING),
    }


def default_output(backend: str, eval_set: bool) -> str:
    """Backend-specific defaults so a jax run and a torchtpu run never overwrite each other."""
    return f"results/06_{'verdicts' if eval_set else 'run'}_{backend}.json"


# ── TPU execution (imports vllm; never reached under --dry-run) ───────────────


def run_band(
    records: list, args: argparse.Namespace, ceiling: int, max_model_len: int, dtype: str, tensor_parallel_size: int
) -> dict:
    """05's run_band with dtype and the chip count as parameters. Keep this a copy.

    The only reason this function exists is 05's hardcoded dtype="bfloat16".
    Every other line follows 05 so tests/test_misreport_runner.py can run
    both under the same fakes and prove the sampling settings are identical.
    There is no backend parameter on purpose: nothing here can branch on it.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    triples = _runner.build_prompts(records, args.single_call)
    prompts = [t[2] for t in triples]
    schema = _schema.VERDICT_SCHEMA if args.single_call else _schema.STATE_VERDICT_SCHEMA
    sampling = SamplingParams(**SAMPLING, structured_outputs=StructuredOutputsParams(json=schema))

    console.print(
        f"[bold]Band {ceiling}[/bold]: {len(records)} records, {len(prompts)} prompts, "
        f"max_model_len={max_model_len}, dtype={dtype}, backend={args.backend}. "
        "First run per shape compiles 20-30 min."
    )
    llm = LLM(model=args.model, tensor_parallel_size=tensor_parallel_size, dtype=dtype, max_model_len=max_model_len)

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


# ── Entry ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    # The guard runs first, before any file is read, so a wrong environment
    # fails in milliseconds. A dry run imports nothing and checks nothing.
    if args.dry_run:
        vllm_distribution = {"name": "unchecked (dry run)", "version": "unchecked (dry run)"}
    else:
        vllm_distribution = require_backend(args.backend, installed_vllm_distributions())

    records = _corpus.load_staged(args.input) if args.staged else _corpus.load_corpus(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        console.print(f"[yellow]No records loaded from {args.input}; nothing to run.[/yellow]")
        return
    report = _runner.load_band_report(args.bands_report)
    ceilings = _runner.selected_ceilings(report, args.band)
    max_model_len = _runner.compute_max_model_len(max(ceilings))
    mode = (
        "dry_run"
        if args.dry_run
        else ("compile_only" if args.compile_only else ("eval_set" if args.eval_set else "benchmark"))
    )
    tensor_parallel_size = resolve_tensor_parallel_size(probe_jax=not args.dry_run)
    label_sources = _labels.summarize(records, staged=args.staged)
    metadata = {
        **_runner.build_run_metadata(args, ceilings, max_model_len, mode),
        **extra_metadata(args, tensor_parallel_size, vllm_distribution, file_sha256(args.bands_report), label_sources),
    }
    console.print(
        f"Labels on these records: {label_sources['statement']}. The judge reads none of them.",
        highlight=False,
        soft_wrap=True,
    )

    plan = []
    for ceiling in ceilings:
        members = _runner.records_for_ceiling(records, report, ceiling)
        triples = _runner.build_prompts(members, args.single_call)
        plan.append({"ceiling": ceiling, "records": len(members), "prompts": len(triples)})

    out_path = Path(args.output or default_output(args.backend, args.eval_set))
    if args.dry_run:
        output = {
            "metadata": metadata,
            "plan": plan,
            "executed": False,
            "note": "dry run: no vllm import, no TPU, no measurement",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        console.print(
            f"[bold]Dry run.[/bold] backend {args.backend} (unchecked), dtype {args.dtype}, mode that WOULD execute: "
            f"{'single-call' if args.single_call else 'six-call'}, bands {ceilings}, "
            f"max_model_len {max_model_len}."
        )
        for entry in plan:
            console.print(f"  band {entry['ceiling']}: {entry['records']} records -> {entry['prompts']} prompts")
        console.print(f"[green]Plan written to {out_path}. No TPU run occurred.[/green]")
        return

    band_results = []
    for ceiling in ceilings:
        members = _runner.records_for_ceiling(records, report, ceiling)
        if not members:
            console.print(f"[yellow]Band {ceiling}: no records, skipping.[/yellow]")
            continue
        band_results.append(run_band(members, args, ceiling, max_model_len, args.dtype, tensor_parallel_size))

    output = {
        "metadata": metadata,
        "bands": [{k: v for k, v in band.items() if k != "responses"} for band in band_results],
    }
    if args.eval_set:
        verdicts, parse_errors = _runner.verdicts_from_band_results(band_results, args.single_call)
        output["verdicts"] = verdicts
        output["metadata"]["parse_errors"] = parse_errors
        output["metadata"]["evaluator"] = "gemma_judge"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    console.print(f"[green]Results written to {out_path}[/green]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 6: trajectory judge on a declared vLLM backend")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_DISTRIBUTIONS),
        required=True,
        help="Which vllm distribution this environment must hold. Recorded, never switched.",
    )
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16", help="LLM dtype; bfloat16 is 05's value")
    # 05's flags, same names and defaults, so the two commands line up.
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
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
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to results/06_run_<backend>.json, or results/06_verdicts_<backend>.json with --eval-set",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
