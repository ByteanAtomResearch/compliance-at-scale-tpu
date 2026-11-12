"""
Module 6: Laptop rehearsal of everything that needs no TPU (Part 3).

`make misreport-rehearsal` runs the laptop-only path of the labs in order and
exits 1 on the first mismatch it finds in any of them:

  1. review files   the committed review files match a rebuild (Lab B)
  2. staged load    05's loader accepts all three staged files (Lab B)
  3. rules baseline 05's rules over the tamper drafts flag exactly one label (Lab B)
  4. plan diff      05's dry run and 06's dry run produce the same plan (Lab D)
  5. parity         two fixture verdict files compare; a tampered one is refused (Lab E)
  6. bridge         a fixture verdict file renders; an unknown state is refused (Lab F)

What the fixtures are. No judge ran here. Steps 5 and 6 need verdict files,
so this script writes two from the tamper drafts' proposed labels, one per
backend name, with one detection removed from the torchtpu side so the
agreement table has a disagreement to find. Their metadata comes from a real
06 dry run, so the hashes are the ones a TPU run would record. Every fixture
is marked `rehearsal_fixture` and lands under results/rehearsal/, which is
gitignored. Nothing this script writes is a result, and nothing it writes may
be copied into published/.

The band reports are stand-ins measured with a whitespace tokenizer, the way
tests/test_misreport_runner.py does it, because the real tokenizer needs a
Hugging Face token. They are marked staged, so freeze.py will not accept them.

Steps 4 to 6 run the same commands the README gives, as subprocesses, and
print each one, so a reader can paste any line into a shell.
"""

import argparse
import copy
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lab05
import yaml
from rich.console import Console

_schema = lab05.load("schema")
_corpus = lab05.load("corpus")
_bands = lab05.load("bands")
_rules = lab05.load("rules_baseline")
_labels = lab05.sibling("labels")

console = Console()

LAB06_DIR = Path(__file__).resolve().parent
DEFAULT_STAGING = LAB06_DIR / "data" / "staging"
DEFAULT_SOURCE = lab05.REPO_ROOT / "sample_data" / "staging" / "candidates_tier2_batch1.jsonl"
DEFAULT_OUT_DIR = lab05.REPO_ROOT / "results" / "rehearsal"
CUT60_FILENAME = "candidates_tier2_cut60.jsonl"
TOPUP14_FILENAME = "candidates_tier2_topup14.jsonl"
STUDENT14_FILENAME = "candidates_tier2_student14.jsonl"
TAMPER_FILENAME = "candidates_tamper_batch1.jsonl"
TAMPER_TIER = "authored_subtle"
MODEL = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
STAND_IN_TOKENIZER = "whitespace stand-in (rehearsal; not the model tokenizer)"
FIXTURE_EVALUATOR = "rehearsal_fixture"
FIXTURE_NOTE = "fixture built from proposed labels by rehearsal.py; no judge ran; never publish"
FIXTURE_POLICY_SOURCE = "operator policy lookup (DEFAULT_ENFORCEMENT_POLICY), not model output"
PLAN_LIMIT = 2
EXIT_REFUSED = 2

# What Lab B's checkpoint records for the tamper drafts.
EXPECTED_STAGED_COUNTS = {CUT60_FILENAME: 60, TOPUP14_FILENAME: 14, STUDENT14_FILENAME: 14, TAMPER_FILENAME: 8}
EXPECTED_RULES_FLAGS = {("traj-t3-tmp-001", "authorization_bypass", 3)}


@dataclass
class StepResult:
    name: str
    problems: list[str] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def expect(self, condition: bool, problem: str) -> None:
        if not condition:
            self.problems.append(problem)


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    """Run one README command from the repo root and echo it first."""
    root = f"{lab05.REPO_ROOT}/"
    shown = " ".join(shlex.quote(part.removeprefix(root)) for part in ["python", *argv])
    console.print(f"  [dim]$ {shown}[/dim]", highlight=False, soft_wrap=True)
    return subprocess.run(
        [sys.executable, *argv], cwd=lab05.REPO_ROOT, capture_output=True, text=True, check=False, timeout=300
    )


def _tail(completed: subprocess.CompletedProcess) -> str:
    text = (completed.stderr or completed.stdout or "").strip()
    return text.splitlines()[-1] if text else "<no output>"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def write_stand_in_band_report(records: list, source: Path, path: Path) -> None:
    """A band report measured by whitespace split. Marked staged so it can never be frozen."""
    lengths = _bands.measure_state_lengths(records, lambda text: len(text.split()))
    report = _bands.banding_report(lengths)
    report["record_lengths"] = {record.id: max(row) for record, row in zip(records, lengths)}
    report["metadata"] = {"corpus": str(source), "staged": True, "tokenizer": STAND_IN_TOKENIZER}
    _write_json(path, report)


# ── Steps 1 to 3: in process, through 05's own functions ──────────────────────


def step_review_files(source: Path, staging: Path) -> StepResult:
    step = StepResult("review files")
    build = _load_sibling("build_review_files")
    problems = build.check(source, staging)
    step.problems.extend(problems)
    step.observed.append(f"{len(build.REVIEW_FILES)} review files compared to a rebuild, {len(problems)} differ")
    return step


def step_staged_load(staging: Path) -> tuple[StepResult, dict[str, list]]:
    step = StepResult("staged load")
    loaded: dict[str, list] = {}
    for filename, expected in EXPECTED_STAGED_COUNTS.items():
        try:
            records = _corpus.load_staged(staging / filename)
        except (OSError, ValueError, _corpus.CorpusIntegrityError) as exc:
            step.problems.append(f"{filename}: 05's loader rejected it: {exc}")
            continue
        loaded[filename] = records
        step.expect(len(records) == expected, f"{filename}: expected {expected} records, loaded {len(records)}")
        adjudicated = [record.id for record in records if record.adjudicated]
        step.expect(not adjudicated, f"{filename}: staged records already adjudicated: {adjudicated[:5]}")
    overlap = {r.id for r in loaded.get(CUT60_FILENAME, [])} & {r.id for r in loaded.get(TOPUP14_FILENAME, [])}
    step.expect(not overlap, f"cut60 and topup14 share ids: {sorted(overlap)[:5]}")
    counts = ", ".join(
        f"{name.removeprefix('candidates_').removesuffix('.jsonl')}={len(r)}" for name, r in loaded.items()
    )
    step.observed.append(f"loaded {counts}; cut60/topup14 overlap {len(overlap)}")
    return step, loaded


def step_rules_baseline(tamper: list, out_dir: Path) -> StepResult:
    step = StepResult("rules baseline")
    verdicts = _rules.evaluate_records(tamper)
    _write_json(
        out_dir / "tamper_rules.json",
        {"metadata": {"evaluator": "rules_baseline", "total_records": len(tamper)}, "verdicts": verdicts},
    )
    flags = {
        (record_id, violation["state"], violation["failed_step_index"])
        for record_id, verdict in verdicts.items()
        for violation in verdict["violations"]
    }
    step.expect(
        flags == EXPECTED_RULES_FLAGS,
        f"rules flagged {sorted(flags)}, Lab B's checkpoint records {sorted(EXPECTED_RULES_FLAGS)}",
    )
    shown = ", ".join(f"{state} on {record_id} at step {index}" for record_id, state, index in sorted(flags))
    step.observed.append(f"{len(verdicts)} records, {len(flags)} flagged: {shown or 'none'}")
    return step


# ── Step 4: the two dry runs, as the README runs them ─────────────────────────


def step_plan_diff(source: Path, out_dir: Path) -> StepResult:
    step = StepResult("plan diff")
    bands_path = out_dir / "stand_in_bands_tier2.json"
    write_stand_in_band_report(_corpus.load_staged(source), source, bands_path)
    plan05_path, plan06_path = out_dir / "plan05.json", out_dir / "plan06.json"
    shared = ["--model", MODEL, "--staged", "--input", str(source), "--bands-report", str(bands_path)]
    shared += ["--limit", str(PLAN_LIMIT), "--dry-run"]
    for path in (plan05_path, plan06_path):
        path.unlink(missing_ok=True)
    run05 = _run(["05_trajectory_eval/batch_trajectory_eval.py", *shared, "--output", str(plan05_path)])
    run06 = _run(
        [f"{LAB06_DIR.name}/run.py", "--backend", "torchtpu", "--dtype", "auto", *shared, "--output", str(plan06_path)]
    )
    for name, completed, path in (("05", run05, plan05_path), ("06", run06, plan06_path)):
        step.expect(completed.returncode == 0, f"{name} dry run exited {completed.returncode}: {_tail(completed)}")
        step.expect(path.exists(), f"{name} dry run wrote no plan at {path}")
    if step.problems:
        return step
    plan05, plan06 = _read_json(plan05_path), _read_json(plan06_path)
    step.expect(plan05["plan"] == plan06["plan"], f"plans differ: 05 {plan05['plan']} vs 06 {plan06['plan']}")
    prompts = sum(entry["prompts"] for entry in plan06["plan"])
    expected_prompts = PLAN_LIMIT * len(_schema.VIOLATION_STATES)
    step.expect(prompts == expected_prompts, f"06 plan has {prompts} prompts, expected {expected_prompts}")
    step.expect(plan06["executed"] is False, "06 dry run claims it executed")
    meta = plan06["metadata"]
    step.expect(
        meta.get("vllm_distribution", {}).get("name") == "unchecked (dry run)",
        f"06 dry run recorded a vllm distribution: {meta.get('vllm_distribution')}",
    )
    for key in ("backend", "dtype", "checkpoint", "band_report_sha256", "git"):
        step.expect(key in meta, f"06 dry-run metadata lacks {key}")
    step.observed.append(
        f"plans {'identical' if plan05['plan'] == plan06['plan'] else 'DIFFERENT'}: "
        f"{len(plan06['plan'])} bands, {prompts} prompts for {PLAN_LIMIT} records"
    )
    return step


# ── Steps 5 and 6: parity and bridge on fixture verdicts ──────────────────────


def fixture_verdicts(records: list) -> dict[str, dict[str, Any]]:
    """One verdict per record, read straight off its proposed labels through 05's own types."""
    verdicts = {}
    for record in records:
        violations = [
            _schema.Violation(
                state=label.state, failed_step_index=label.step_index, evidence="rehearsal fixture", confidence=1.0
            )
            for label in record.proposed_labels
        ]
        verdict = _schema.Verdict(violations=violations, recommended_enforcement=_schema.derive_enforcement(violations))
        _schema.validate_verdict(verdict)
        verdicts[record.id] = _schema.verdict_to_dict(verdict)
    return verdicts


def plant_disagreement(verdicts: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str, str]:
    """Drop the first detection of the first flagged record. Returns the copy, the id and the state."""
    planted = copy.deepcopy(verdicts)
    record_id = next(rid for rid in sorted(planted) if planted[rid]["violations"])
    state = planted[record_id]["violations"].pop(0)["state"]
    return planted, record_id, state


def write_fixture_runs(tamper: list, staging: Path, out_dir: Path, step: StepResult) -> dict[str, Any] | None:
    """Two verdict files that differ in backend and in one planted detection."""
    tamper_path = staging / TAMPER_FILENAME
    bands_path = out_dir / "stand_in_bands_tamper.json"
    write_stand_in_band_report(tamper, tamper_path, bands_path)
    plan_path = out_dir / "plan06_tamper.json"
    plan_path.unlink(missing_ok=True)
    completed = _run(
        [
            f"{LAB06_DIR.name}/run.py",
            *("--backend", "torchtpu", "--model", MODEL, "--staged", "--input", str(tamper_path)),
            *("--bands-report", str(bands_path), "--dry-run", "--output", str(plan_path)),
        ]
    )
    if completed.returncode != 0 or not plan_path.exists():
        step.problems.append(f"06 dry run over the tamper drafts exited {completed.returncode}: {_tail(completed)}")
        return None
    plan = _read_json(plan_path)
    base_meta = {
        **plan["metadata"],
        "mode": "eval_set",
        "parse_errors": 0,
        "evaluator": FIXTURE_EVALUATOR,
        "note": FIXTURE_NOTE,
    }
    left = fixture_verdicts(tamper)
    right, planted_id, planted_state = plant_disagreement(left)
    bands = [{"ceiling": e["ceiling"], "records": e["records"], "prompts": e["prompts"]} for e in plan["plan"]]
    paths = {}
    for backend, verdicts in (("jax", left), ("torchtpu", right)):
        paths[backend] = out_dir / f"fixture_verdicts_{backend}.json"
        _write_json(
            paths[backend], {"metadata": {**base_meta, "backend": backend}, "bands": bands, "verdicts": verdicts}
        )
    return {
        "paths": paths,
        "labels": tamper_path,
        "right": right,
        "planted_id": planted_id,
        "planted_state": planted_state,
        "n": len(tamper),
    }


def step_parity(fixtures: dict[str, Any] | None, out_dir: Path, step: StepResult) -> StepResult:
    if fixtures is None:
        return step
    jax_path, torchtpu_path = fixtures["paths"]["jax"], fixtures["paths"]["torchtpu"]
    parity_path = out_dir / "fixture_parity.json"
    parity_path.unlink(missing_ok=True)
    sides = ["--left", str(jax_path), "--left-name", "jax", "--right", str(torchtpu_path), "--right-name", "torchtpu"]
    completed = _run(
        [
            f"{LAB06_DIR.name}/parity.py",
            *sides,
            *("--labels", str(fixtures["labels"]), "--allow-staged", "--output", str(parity_path)),
        ]
    )
    step.expect(completed.returncode == 0, f"parity exited {completed.returncode}: {_tail(completed)}")
    if completed.returncode == 0 and parity_path.exists():
        report = _read_json(parity_path)
        n, states = fixtures["n"], _schema.VIOLATION_STATES
        planted_id, planted_state = fixtures["planted_id"], fixtures["planted_state"]
        step.expect(list(report["agreement"]) == list(states), f"states reported: {list(report['agreement'])}")
        for state in states:
            entry = report["agreement"].get(state, {})
            want = [planted_id] if state == planted_state else []
            step.expect(entry.get("n") == n, f"{state}: n={entry.get('n')}, expected {n}")
            step.expect(
                entry.get("disagree_ids") == want, f"{state}: disagree_ids {entry.get('disagree_ids')}, expected {want}"
            )
        overall = report["overall"]
        decisions = n * len(states)
        step.expect(overall["decisions"] == decisions, f"{overall['decisions']} decisions, expected {decisions}")
        step.expect(overall["agree"] == decisions - 1, f"{overall['agree']} agree, expected {decisions - 1}")
        # Nobody has adjudicated the staged drafts, and the report has to say so in labels.py's words.
        tamper_sources = report["labels"]["sources"]["by_tier"].get(TAMPER_TIER, {}).get("sources")
        step.expect(
            tamper_sources == {_labels.LABELS_PROPOSED: n},
            f"the staged drafts' labels were reported as {tamper_sources}, expected {_labels.LABELS_PROPOSED!r}",
        )
        step.expect(report["labels"]["tier_notes"][TAMPER_TIER]["counts_only"], "eight records were scored with rates")
        step.observed.append(
            f"{overall['agree']} of {overall['decisions']} decisions agree; "
            f"the planted disagreement is {planted_state} on {planted_id}"
        )
    else:
        step.expect(parity_path.exists(), f"parity wrote no report at {parity_path}")

    # The guard: one changed hash must be refused with exit 2 and nothing written.
    tampered_path, refused_path = out_dir / "fixture_tampered.json", out_dir / "parity_should_not_exist.json"
    tampered = _read_json(torchtpu_path)
    tampered["metadata"]["prompt_hash"] = "tampered"
    _write_json(tampered_path, tampered)
    refused_path.unlink(missing_ok=True)
    refused = _run(
        [
            f"{LAB06_DIR.name}/parity.py",
            *("--left", str(jax_path), "--right", str(tampered_path), "--output", str(refused_path)),
        ]
    )
    step.expect(refused.returncode == EXIT_REFUSED, f"tampered parity exited {refused.returncode}, expected 2")
    step.expect("prompt_hash" in refused.stderr, "the refusal did not name prompt_hash")
    step.expect(not refused_path.exists(), f"a refused comparison still wrote {refused_path}")
    step.observed.append(f"tampered prompt_hash: exit {refused.returncode}, report written: {refused_path.exists()}")
    return step


def step_bridge(fixtures: dict[str, Any] | None, out_dir: Path) -> StepResult:
    step = StepResult("bridge")
    if fixtures is None:
        step.problems.append("no fixture verdicts to render; see the parity step")
        return step
    torchtpu_path = fixtures["paths"]["torchtpu"]
    md_path, yaml_path = out_dir / "fixture_report.md", out_dir / "fixture_report.yaml"
    for path in (md_path, yaml_path):
        path.unlink(missing_ok=True)
    completed = _run(
        [
            f"{LAB06_DIR.name}/bridge.py",
            *("--verdicts", str(torchtpu_path), "--markdown", str(md_path), "--yaml", str(yaml_path)),
        ]
    )
    step.expect(completed.returncode == 0, f"bridge exited {completed.returncode}: {_tail(completed)}")
    step.expect(md_path.exists(), f"bridge wrote no Markdown at {md_path}")
    if completed.returncode == 0 and yaml_path.exists():
        report = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        step.expect(list(report) == ["rai_evaluation", "policy", "provenance"], f"top-level keys are {list(report)}")
        titles = [state.replace("_", " ").title() for state in _schema.VIOLATION_STATES]
        sections = report.get("rai_evaluation", {})
        step.expect(list(sections) == titles, f"sections are {list(sections)}, expected {titles}")
        flagged: dict[str, list[str]] = {state: [] for state in _schema.VIOLATION_STATES}
        for record_id, verdict in fixtures["right"].items():
            for state in {violation["state"] for violation in verdict["violations"]}:
                flagged[state].append(record_id)
        statuses = []
        for state, title in zip(_schema.VIOLATION_STATES, titles):
            section = sections.get(title, {})
            want_status = "FAIL" if flagged[state] else "PASS"
            step.expect(
                section.get("status") == want_status, f"{title}: status {section.get('status')}, expected {want_status}"
            )
            step.expect(
                section.get("flagged_records") == sorted(flagged[state]),
                f"{title}: flagged {section.get('flagged_records')}, expected {sorted(flagged[state])}",
            )
            step.expect("recommended_enforcement" not in section, f"{title}: enforcement leaked into the section")
            statuses.append(section.get("status"))
        step.expect(
            set(report.get("policy", {}).get("recommended_enforcement", {})) == set(fixtures["right"]),
            "the policy block does not cover every record",
        )
        step.expect(
            report.get("provenance", {}).get("backend") == "torchtpu",
            f"provenance backend is {report.get('provenance', {}).get('backend')}",
        )
        step.observed.append(f"{len(sections)} sections: {statuses.count('FAIL')} FAIL, {statuses.count('PASS')} PASS")
    else:
        step.expect(yaml_path.exists(), f"bridge wrote no YAML at {yaml_path}")

    # The guard: a state 05 does not know must be refused with exit 2 and nothing written.
    bad_path, bad_yaml = out_dir / "fixture_bad_state.json", out_dir / "bad_should_not_exist.yaml"
    bad = _read_json(torchtpu_path)
    first = next(iter(bad["verdicts"]))
    bad["verdicts"][first]["violations"].append(
        {"state": "vibes", "failed_step_index": 0, "evidence": "x", "confidence": 0.5}
    )
    _write_json(bad_path, bad)
    bad_yaml.unlink(missing_ok=True)
    refused = _run([f"{LAB06_DIR.name}/bridge.py", "--verdicts", str(bad_path), "--yaml", str(bad_yaml)])
    step.expect(refused.returncode == EXIT_REFUSED, f"bad-state bridge exited {refused.returncode}, expected 2")
    step.expect("vibes" in refused.stderr, "the refusal did not name the unknown state")
    step.expect(not bad_yaml.exists(), f"a refused bridge still wrote {bad_yaml}")
    step.observed.append(f"unknown state 'vibes': exit {refused.returncode}, report written: {bad_yaml.exists()}")
    return step


# ── Entry ─────────────────────────────────────────────────────────────────────


def _load_sibling(name: str):
    """Load a sibling 06 script by path, the way the tests do, without touching sys.path."""
    import importlib.util

    module_name = f"misreport_rehearsal_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(LAB06_DIR / f"{name}.py"))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}.py from {LAB06_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def rehearse(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    staging: str | Path = DEFAULT_STAGING,
    source: str | Path = DEFAULT_SOURCE,
) -> list[StepResult]:
    """Run all six steps and return every result. Later steps still run after a failure."""
    out_dir, staging, source = Path(out_dir).resolve(), Path(staging).resolve(), Path(source).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [step_review_files(source, staging)]
    staged_step, loaded = step_staged_load(staging)
    results.append(staged_step)
    tamper = loaded.get(TAMPER_FILENAME)
    if tamper:
        results.append(step_rules_baseline(tamper, out_dir))
    else:
        results.append(StepResult("rules baseline", ["the tamper drafts did not load; see the staged load step"]))
    results.append(step_plan_diff(source, out_dir))
    parity_step = StepResult("parity")
    fixtures = write_fixture_runs(tamper, staging, out_dir, parity_step) if tamper else None
    if not tamper:
        parity_step.problems.append("no tamper drafts to build fixture verdicts from")
    results.append(step_parity(fixtures, out_dir, parity_step))
    results.append(step_bridge(fixtures, out_dir))
    return results


def print_results(results: list[StepResult], out_dir: Path) -> None:
    for number, step in enumerate(results, start=1):
        status = "[green]ok[/green]" if step.ok else "[red]MISMATCH[/red]"
        console.print(f"{number}. {step.name}: {status}", highlight=False)
        for line in step.observed:
            console.print(f"     {line}", highlight=False, soft_wrap=True)
        for problem in step.problems:
            console.print(f"     [red]{problem}[/red]", highlight=False, soft_wrap=True)
    failed = [step.name for step in results if not step.ok]
    if failed:
        console.print(f"[red]Rehearsal FAILED: {len(failed)} of {len(results)} steps mismatched: {failed}[/red]")
    else:
        console.print(
            f"[green]Rehearsal passed: {len(results)} of {len(results)} steps, no TPU, no judge.[/green] "
            f"Fixtures are under {out_dir} and are not results.",
            highlight=False,
            soft_wrap=True,
        )


def main(args: argparse.Namespace) -> None:
    results = rehearse(args.out_dir, args.staging, args.source)
    print_results(results, Path(args.out_dir))
    if any(not step.ok for step in results):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 6: rehearse every laptop-only step; exit 1 on any mismatch")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where plans and fixtures go (gitignored)")
    parser.add_argument("--staging", default=str(DEFAULT_STAGING), help="06's staged review files")
    parser.add_argument(
        "--source", default=str(DEFAULT_SOURCE), help="Staged tier-2 batch the review files are cut from"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
