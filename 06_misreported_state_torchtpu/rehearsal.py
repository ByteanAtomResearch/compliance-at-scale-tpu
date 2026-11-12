"""
Module 6: Laptop rehearsal of everything that needs no TPU (Part 3).

`make misreport-rehearsal` runs every step a laptop can run, in order, and
exits 1 when any of them finds a mismatch. Steps 1 to 6 cover the staged
files and the tools. Steps 7 to 11 are a dry run of the student track:

  1. review files    the committed review files and the student sample match a rebuild
  2. staged load     05's loader accepts all four staged files
  3. rules baseline  05's rules over the tamper drafts flag exactly one label
  4. plan diff       05's dry run and 06's dry run produce the same plan
  5. parity          two fixture verdict files compare; a tampered one is refused
  6. bridge          a fixture verdict file renders; an unknown state is refused
  7. student set     adjudicate.py takes its keys from a pipe, then the 22-record
                     eval set builds; with nothing decided it is refused (student steps 3 and 4)
  8. student plan    run.py plans the eval set on both backends, and on the E4B
                     fallback model, without loading anything (student steps 5 and 7)
  9. three readers   rules, a fixture judge and the labels, case by case; a
                     missing verdict is refused (student step 6)
 10. student parity  the two fixture backends compare with labels; a dtype
                     difference and a model difference are each refused (student step 7)
 11. student bridge  the report renders and carries the label statement (student step 8)

What the fixtures are. No judge ran here. The steps that need verdict files
get them from this script: one verdict per record, read off the record's
labels, one file per backend name, with one detection removed from the
torchtpu side so the tables have a disagreement to find. Their metadata comes
from a real 06 dry run, so the hashes are the ones a TPU run would record.
Every fixture is marked `rehearsal_fixture` and lands under
results/rehearsal/, which is gitignored. Nothing this script writes is a
result, and nothing it writes may be copied into published/.

What the fixture adjudication is. Step 7 runs the real adjudicate.py with the
adjudicator name `rehearsal-fixture` and one key per record on stdin: accept
where a draft proposes a violation, clean where it proposes none. That makes
an adjudicated file with the shape yours will have, so the later steps read
"student-adjudicated (by rehearsal-fixture)". No person reviewed anything.
It lands under results/rehearsal/student/, never under results/student/,
which is where your own review goes.

The band reports are stand-ins measured with a whitespace tokenizer, the way
tests/test_misreport_runner.py does it, because the real tokenizer needs a
Hugging Face token. They are marked staged, so freeze.py will not accept them.

Steps 4 to 11 run the same commands the README gives, as subprocesses, and
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
TAMPER_PREFIX = "traj-t3-tmp-"
MODEL = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
FALLBACK_MODEL = "google/gemma-4-E4B-it"
SAMPLE_TIER = "authored_clean"
STUDENT_SUBDIR = "student"
FIXTURE_ADJUDICATOR = "rehearsal-fixture"
BACKENDS = ("jax", "torchtpu")
RATE_WORDS = ("rate", "recall", "precision", "kappa", "wilson", "specificity")
STAND_IN_TOKENIZER = "whitespace stand-in (rehearsal; not the model tokenizer)"
FIXTURE_EVALUATOR = "rehearsal_fixture"
FIXTURE_NOTE = "fixture built from proposed labels by rehearsal.py; no judge ran; never publish"
STUDENT_FIXTURE_NOTE = "fixture built from the eval set's labels by rehearsal.py; no judge ran; never publish"
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


def _run(argv: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run one README command from the repo root and echo it first. stdin stands in for a keyboard."""
    root = f"{lab05.REPO_ROOT}/"
    shown = " ".join(shlex.quote(part.removeprefix(root)) for part in ["python", *argv])
    if stdin is not None:
        shown = f"printf {shlex.quote(stdin.encode('unicode_escape').decode())} | {shown}"
    console.print(f"  [dim]$ {shown}[/dim]", highlight=False, soft_wrap=True)
    return subprocess.run(
        [sys.executable, *argv],
        cwd=lab05.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        input=stdin,
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


def fixture_verdicts(records: list, use_decisions: bool = False) -> dict[str, dict[str, Any]]:
    """One verdict per record, read straight off its labels through 05's own types.

    The proposals by default. With use_decisions, whatever a scorer would
    hold the record to: an adjudicated record's decision, a staged record's
    proposal, nothing for a rejected one.
    """
    verdicts = {}
    for record in records:
        labels = (_labels.scoring_labels(record) or []) if use_decisions else record.proposed_labels
        violations = [
            _schema.Violation(
                state=label.state, failed_step_index=label.step_index, evidence="rehearsal fixture", confidence=1.0
            )
            for label in labels
        ]
        verdict = _schema.Verdict(violations=violations, recommended_enforcement=_schema.derive_enforcement(violations))
        _schema.validate_verdict(verdict)
        verdicts[record.id] = _schema.verdict_to_dict(verdict)
    return verdicts


def plant_disagreement(
    verdicts: dict[str, dict[str, Any]], prefix: str = ""
) -> tuple[dict[str, dict[str, Any]], str, str]:
    """Drop the first detection of the first flagged record whose id starts with prefix.

    Returns the copy, the id and the state.
    """
    planted = copy.deepcopy(verdicts)
    record_id = next(rid for rid in sorted(planted) if rid.startswith(prefix) and planted[rid]["violations"])
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


# ── Steps 7 to 11: the student track, dry ─────────────────────────────────────


def _student_dir(out_dir: Path) -> Path:
    return out_dir / STUDENT_SUBDIR


def _rate_keys(node: Any) -> list[str]:
    """Every dict key in a JSON document that names a rate. The per-case file must have none."""
    if isinstance(node, dict):
        own = [key for key in node if any(word in key.lower() for word in RATE_WORDS)]
        return sorted(set(own + [key for value in node.values() for key in _rate_keys(value)]))
    if isinstance(node, list):
        return sorted({key for value in node for key in _rate_keys(value)})
    return []


def _tier_sources(summary: dict[str, Any], tier: str) -> dict[str, int] | None:
    return summary.get("by_tier", {}).get(tier, {}).get("sources")


def step_student_set(tamper: list, staging: Path, out_dir: Path) -> tuple[StepResult, Path | None]:
    """Student steps 3 and 4: review the eight, then build the eval set from staged files."""
    step = StepResult("student set")
    student_dir = _student_dir(out_dir)
    student_dir.mkdir(parents=True, exist_ok=True)
    decisions, rejections = student_dir / "tamper_labels.jsonl", student_dir / "tamper_rejections.jsonl"
    eval_set = student_dir / "eval_set.jsonl"
    for path in (decisions, rejections, eval_set):
        path.unlink(missing_ok=True)
    tamper_path = staging / TAMPER_FILENAME
    build = [
        f"{LAB06_DIR.name}/student_set.py",
        *("--tamper", str(tamper_path), "--sample", str(staging / STUDENT14_FILENAME)),
        *("--decisions", str(decisions), "--rejections", str(rejections), "--output", str(eval_set)),
    ]

    # The guard: with nothing decided the build is refused with exit 2 and nothing written.
    refused = _run(build)
    step.expect(refused.returncode == EXIT_REFUSED, f"an unreviewed build exited {refused.returncode}, expected 2")
    step.expect(f"{len(tamper)} of {len(tamper)}" in refused.stderr, "the refusal did not count the undecided records")
    step.expect(not eval_set.exists(), f"a refused build still wrote {eval_set}")
    step.observed.append(f"nothing decided: exit {refused.returncode}, eval set written: {eval_set.exists()}")

    # adjudicate.py as the student runs it, with one key per record on stdin.
    keys = "".join("a\n" if record.proposed_labels else "c\n" for record in tamper)
    adjudicated = _run(
        [
            "05_trajectory_eval/adjudicate.py",
            *("--adjudicator", FIXTURE_ADJUDICATOR, "--staged", str(tamper_path)),
            *("--corpus", str(decisions), "--rejections", str(rejections)),
        ],
        stdin=keys,
    )
    step.expect(adjudicated.returncode == 0, f"adjudicate.py exited {adjudicated.returncode}: {_tail(adjudicated)}")
    promoted = _corpus.load_corpus(decisions) if adjudicated.returncode == 0 else []
    step.expect(len(promoted) == len(tamper), f"adjudicate.py promoted {len(promoted)} of {len(tamper)} records")
    step.observed.append(
        f"{len(keys.split())} keys piped to adjudicate.py as {FIXTURE_ADJUDICATOR}: {len(promoted)} promoted"
    )

    built = _run(build)
    step.expect(built.returncode == 0, f"student_set.py exited {built.returncode}: {_tail(built)}")
    if built.returncode != 0 or not eval_set.exists():
        step.expect(eval_set.exists(), f"student_set.py wrote no eval set at {eval_set}")
        return step, None
    records = _corpus.load_staged(eval_set)
    summary = _labels.summarize(records, staged=True)
    sample_n = EXPECTED_STAGED_COUNTS[STUDENT14_FILENAME]
    step.expect(len(records) == len(tamper) + sample_n, f"eval set has {len(records)} records")
    step.expect(
        _tier_sources(summary, TAMPER_TIER) == {_labels.LABELS_STUDENT: len(tamper)},
        f"the eight are labelled {_tier_sources(summary, TAMPER_TIER)}, expected {_labels.LABELS_STUDENT!r}",
    )
    step.expect(
        summary["by_tier"].get(TAMPER_TIER, {}).get("adjudicators") == [FIXTURE_ADJUDICATOR],
        "the fixture adjudication does not carry the fixture's name",
    )
    step.expect(
        _tier_sources(summary, SAMPLE_TIER) == {_labels.LABELS_TEMPLATE: sample_n},
        f"the sample is labelled {_tier_sources(summary, SAMPLE_TIER)}, expected {_labels.LABELS_TEMPLATE!r}",
    )
    step.observed.append(f"{len(records)} records. Labels: {summary['statement']}")
    return step, eval_set


def step_student_plan(eval_set: Path | None, out_dir: Path) -> tuple[StepResult, dict[str, Any] | None]:
    """Student steps 5 and 7, dry: both backends, then both again on the E4B fallback model."""
    step = StepResult("student plan")
    if eval_set is None:
        step.problems.append("no student eval set to plan; see the student set step")
        return step, None
    student_dir = _student_dir(out_dir)
    records = _corpus.load_staged(eval_set)
    bands_path = student_dir / "stand_in_bands.json"
    write_stand_in_band_report(records, eval_set, bands_path)
    plans: dict[str, dict[str, Any]] = {}
    for tag, model in (("26b", MODEL), ("e4b", FALLBACK_MODEL)):
        for backend in BACKENDS:
            plan_path = student_dir / f"plan_{tag}_{backend}.json"
            plan_path.unlink(missing_ok=True)
            completed = _run(
                [
                    f"{LAB06_DIR.name}/run.py",
                    *("--backend", backend, "--model", model, "--staged", "--input", str(eval_set)),
                    *("--bands-report", str(bands_path), "--dry-run", "--output", str(plan_path)),
                ]
            )
            if completed.returncode != 0 or not plan_path.exists():
                step.problems.append(f"{tag} {backend} dry run exited {completed.returncode}: {_tail(completed)}")
                continue
            plans[f"{tag}_{backend}"] = _read_json(plan_path)
    if len(plans) != 2 * len(BACKENDS):
        return step, None

    prompts = len(records) * len(_schema.VIOLATION_STATES)
    tamper_n = sum(1 for record in records if record.tier == TAMPER_TIER)
    for name, plan in plans.items():
        tag, backend = name.split("_")
        meta = plan["metadata"]
        planned = sum(entry["prompts"] for entry in plan["plan"])
        step.expect(planned == prompts, f"{name}: {planned} prompts planned, expected {prompts}")
        step.expect(plan["executed"] is False, f"{name}: the dry run claims it executed")
        step.expect(meta.get("backend") == backend, f"{name}: recorded backend {meta.get('backend')}")
        step.expect(
            meta.get("model") == (MODEL if tag == "26b" else FALLBACK_MODEL), f"{name}: model {meta.get('model')}"
        )
        sources = meta.get("label_sources") or {}
        step.expect(sources.get("read_by_judge") is False, f"{name}: the run log does not say labels are unread")
        step.expect(
            _tier_sources(sources, TAMPER_TIER) == {_labels.LABELS_STUDENT: tamper_n},
            f"{name}: the run log labels the eight {_tier_sources(sources, TAMPER_TIER)}",
        )
        step.expect(
            _tier_sources(sources, SAMPLE_TIER) == {_labels.LABELS_TEMPLATE: len(records) - tamper_n},
            f"{name}: the run log labels the sample {_tier_sources(sources, SAMPLE_TIER)}",
        )
    step.expect(plans["26b_jax"]["plan"] == plans["26b_torchtpu"]["plan"], "the two backends plan different work")
    bands = len(plans["26b_torchtpu"]["plan"])
    step.observed.append(
        f"{len(records)} records -> {prompts} prompts in {bands} band(s); the same plan on jax and torchtpu"
    )
    step.observed.append(f"E4B fallback ({FALLBACK_MODEL}): the same {prompts} prompts on both backends")
    return step, {"plans": plans, "records": records, "eval_set": eval_set}


def write_student_fixture_runs(student: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Verdict files for the eval set on both backend names and both models. One planted miss on torchtpu."""
    student_dir = _student_dir(out_dir)
    left = fixture_verdicts(student["records"], use_decisions=True)
    right, planted_id, planted_state = plant_disagreement(left, prefix=TAMPER_PREFIX)
    paths: dict[str, Path] = {}
    for tag in ("26b", "e4b"):
        for backend, verdicts in (("jax", left), ("torchtpu", right)):
            plan = student["plans"][f"{tag}_{backend}"]
            metadata = {
                **plan["metadata"],
                "mode": "eval_set",
                "parse_errors": 0,
                "evaluator": FIXTURE_EVALUATOR,
                "note": STUDENT_FIXTURE_NOTE,
            }
            bands = [{"ceiling": e["ceiling"], "records": e["records"], "prompts": e["prompts"]} for e in plan["plan"]]
            name = backend if tag == "26b" else f"{tag}_{backend}"
            paths[name] = student_dir / f"fixture_verdicts_{name}.json"
            _write_json(paths[name], {"metadata": metadata, "bands": bands, "verdicts": verdicts})
    return {"paths": paths, "right": right, "planted_id": planted_id, "planted_state": planted_state}


def step_three_readers(student: dict[str, Any] | None, fixtures: dict[str, Any] | None, out_dir: Path) -> StepResult:
    """Student step 6: rules, judge and labels over the same records, case by case."""
    step = StepResult("three readers")
    if student is None or fixtures is None:
        step.problems.append("no student eval set or fixture verdicts; see the student set and student plan steps")
        return step
    student_dir, eval_set, records = _student_dir(out_dir), student["eval_set"], student["records"]
    rules_path, readers_path = student_dir / "rules.json", student_dir / "readers.json"
    for path in (rules_path, readers_path):
        path.unlink(missing_ok=True)
    rules = _run(
        ["05_trajectory_eval/rules_baseline.py", "--staged", "--input", str(eval_set), "--output", str(rules_path)]
    )
    step.expect(rules.returncode == 0 and rules_path.exists(), f"rules baseline exited {rules.returncode}")
    judge_path = fixtures["paths"]["torchtpu"]
    compare = [
        f"{LAB06_DIR.name}/compare_readers.py",
        *("--records", str(eval_set), "--allow-staged", "--rules", str(rules_path)),
    ]
    completed = _run([*compare, "--judge", str(judge_path), "--output", str(readers_path)])
    step.expect(completed.returncode == 0, f"compare_readers exited {completed.returncode}: {_tail(completed)}")
    if completed.returncode == 0 and readers_path.exists():
        report = _read_json(readers_path)
        tamper_n = sum(1 for record in records if record.tier == TAMPER_TIER)
        sample_n = len(records) - tamper_n
        planted_id, planted_state = fixtures["planted_id"], fixtures["planted_state"]
        step.expect(len(report["cases"]) == len(records), f"{len(report['cases'])} rows for {len(records)} records")
        tamper, sample = report["counts"].get(TAMPER_TIER, {}), report["counts"].get(SAMPLE_TIER, {})
        step.expect(tamper.get("records") == tamper_n, f"tamper table has {tamper.get('records')} rows")
        step.expect(sample.get("records") == sample_n, f"sample table has {sample.get('records')} rows")
        step.expect(
            tamper.get("rules_flagged_any") == len(EXPECTED_RULES_FLAGS),
            f"the rules flagged {tamper.get('rules_flagged_any')} tamper records, expected {len(EXPECTED_RULES_FLAGS)}",
        )
        step.expect(
            tamper.get("judge_same_states_as_labels") == tamper_n - 1,
            f"the fixture judge matched the labels on {tamper.get('judge_same_states_as_labels')} tamper records",
        )
        row = next((case for case in report["cases"] if case["id"] == planted_id), {})
        missed = (row.get("judge_vs_labels") or {}).get("missed")
        step.expect(missed == [planted_state], f"{planted_id}: the judge column shows missed={missed}")
        sources = report["metadata"]["label_sources"]
        step.expect(
            _tier_sources(sources, TAMPER_TIER) == {_labels.LABELS_STUDENT: tamper_n},
            f"the tamper table names its labels as {_tier_sources(sources, TAMPER_TIER)}",
        )
        step.expect(
            _tier_sources(sources, SAMPLE_TIER) == {_labels.LABELS_TEMPLATE: sample_n},
            f"the sample table names its labels as {_tier_sources(sources, SAMPLE_TIER)}",
        )
        rates = _rate_keys(report)
        step.expect(not rates, f"the per-case file carries rate fields: {rates}")
        step.observed.append(
            f"{tamper_n} tamper rows: the rules flag {tamper.get('rules_flagged_any')}, the fixture judge names the "
            f"labelled states on {tamper.get('judge_same_states_as_labels')} (planted miss: {planted_state} on "
            f"{planted_id}); {sample_n} sample rows; rate fields in the file: {len(rates)}"
        )
    else:
        step.expect(readers_path.exists(), f"compare_readers wrote nothing at {readers_path}")

    # The guard: a record with no judge verdict must be refused with exit 2 and nothing written.
    short_path, refused_path = (
        student_dir / "fixture_missing_verdict.json",
        student_dir / "readers_should_not_exist.json",
    )
    short = _read_json(judge_path)
    dropped = sorted(short["verdicts"])[-1]
    del short["verdicts"][dropped]
    _write_json(short_path, short)
    refused_path.unlink(missing_ok=True)
    refused = _run([*compare, "--judge", str(short_path), "--output", str(refused_path)])
    step.expect(refused.returncode == EXIT_REFUSED, f"a missing verdict exited {refused.returncode}, expected 2")
    step.expect(dropped in refused.stderr, "the refusal did not name the record with no verdict")
    step.expect(not refused_path.exists(), f"a refused comparison still wrote {refused_path}")
    step.observed.append(f"one verdict missing: exit {refused.returncode}, report written: {refused_path.exists()}")
    return step


def step_student_parity(student: dict[str, Any] | None, fixtures: dict[str, Any] | None, out_dir: Path) -> StepResult:
    """Student step 7: the two backends on the eval set, with labels, plus the two refusals the step relies on."""
    step = StepResult("student parity")
    if student is None or fixtures is None:
        step.problems.append("no student eval set or fixture verdicts; see the student set and student plan steps")
        return step
    student_dir, eval_set, records = _student_dir(out_dir), student["eval_set"], student["records"]
    paths = fixtures["paths"]
    decisions = len(records) * len(_schema.VIOLATION_STATES)
    labelled = ["--labels", str(eval_set), "--allow-staged"]

    def sides(left: Path, right: Path) -> list[str]:
        return ["--left", str(left), "--left-name", "jax", "--right", str(right), "--right-name", "torchtpu"]

    parity_path = student_dir / "parity.json"
    parity_path.unlink(missing_ok=True)
    completed = _run(
        [
            f"{LAB06_DIR.name}/parity.py",
            *sides(paths["jax"], paths["torchtpu"]),
            *labelled,
            "--output",
            str(parity_path),
        ]
    )
    step.expect(completed.returncode == 0, f"parity exited {completed.returncode}: {_tail(completed)}")
    if completed.returncode == 0 and parity_path.exists():
        report = _read_json(parity_path)
        planted_id, planted_state = fixtures["planted_id"], fixtures["planted_state"]
        overall = report["overall"]
        step.expect(overall["decisions"] == decisions, f"{overall['decisions']} decisions, expected {decisions}")
        step.expect(overall["agree"] == decisions - 1, f"{overall['agree']} agree, expected {decisions - 1}")
        disagree = report["agreement"].get(planted_state, {}).get("disagree_ids")
        step.expect(disagree == [planted_id], f"{planted_state}: disagree_ids {disagree}, expected {[planted_id]}")
        labels = report["labels"]
        tamper_n = sum(1 for record in records if record.tier == TAMPER_TIER)
        step.expect(
            _tier_sources(labels["sources"], TAMPER_TIER) == {_labels.LABELS_STUDENT: tamper_n},
            f"parity names the eight's labels as {_tier_sources(labels['sources'], TAMPER_TIER)}",
        )
        step.expect(
            _tier_sources(labels["sources"], SAMPLE_TIER) == {_labels.LABELS_TEMPLATE: len(records) - tamper_n},
            f"parity names the sample's labels as {_tier_sources(labels['sources'], SAMPLE_TIER)}",
        )
        for tier in (TAMPER_TIER, SAMPLE_TIER):
            step.expect(labels["tier_notes"][tier]["counts_only"], f"tier {tier} was scored with rates")
            rated = [
                state
                for state, entry in labels["tiers"][tier].items()
                if entry["left"]["recall"] is not None or entry["right"]["precision"] is not None
            ]
            step.expect(not rated, f"tier {tier} carries rates for {rated}")
        step.observed.append(
            f"{overall['agree']} of {overall['decisions']} decisions agree; the planted disagreement is "
            f"{planted_state} on {planted_id}; both tiers reported as counts"
        )
        step.observed.append(f"labels: {labels['source']}")
    else:
        step.expect(parity_path.exists(), f"parity wrote no report at {parity_path}")

    # Guard 1: the two backends must run with the same --dtype, or parity refuses.
    auto_path, refused_path = student_dir / "fixture_dtype_auto.json", student_dir / "parity_should_not_exist.json"
    auto = _read_json(paths["torchtpu"])
    auto["metadata"]["dtype"] = "auto"
    _write_json(auto_path, auto)
    refused_path.unlink(missing_ok=True)
    refused = _run([f"{LAB06_DIR.name}/parity.py", *sides(paths["jax"], auto_path), "--output", str(refused_path)])
    step.expect(refused.returncode == EXIT_REFUSED, f"a dtype difference exited {refused.returncode}, expected 2")
    step.expect("dtype" in refused.stderr, "the refusal did not name dtype")
    step.expect(not refused_path.exists(), f"a refused comparison still wrote {refused_path}")
    step.observed.append(f"dtype bfloat16 vs auto: exit {refused.returncode}, report written: {refused_path.exists()}")

    # Guard 2, the E4B fallback: a 26B run is never compared with an E4B run, and E4B against E4B compares.
    mixed = _run(
        [f"{LAB06_DIR.name}/parity.py", *sides(paths["e4b_jax"], paths["torchtpu"]), "--output", str(refused_path)]
    )
    step.expect(mixed.returncode == EXIT_REFUSED, f"E4B against 26B exited {mixed.returncode}, expected 2")
    step.expect("model:" in mixed.stderr, "the refusal did not name the model")
    step.expect(not refused_path.exists(), f"a refused comparison still wrote {refused_path}")
    e4b_path = student_dir / "parity_e4b.json"
    e4b_path.unlink(missing_ok=True)
    fallback = _run(
        [
            f"{LAB06_DIR.name}/parity.py",
            *sides(paths["e4b_jax"], paths["e4b_torchtpu"]),
            *labelled,
            *("--output", str(e4b_path)),
        ]
    )
    step.expect(fallback.returncode == 0 and e4b_path.exists(), f"E4B against E4B exited {fallback.returncode}")
    step.observed.append(
        f"E4B fallback: E4B jax vs 26B torchtpu exit {mixed.returncode}; E4B vs E4B exit {fallback.returncode}"
    )
    return step


def step_student_bridge(student: dict[str, Any] | None, fixtures: dict[str, Any] | None, out_dir: Path) -> StepResult:
    """Student step 8: the rai-checklist-cli shapes, with the label statement in the provenance block."""
    step = StepResult("student bridge")
    if student is None or fixtures is None:
        step.problems.append("no student eval set or fixture verdicts; see the student set and student plan steps")
        return step
    student_dir = _student_dir(out_dir)
    md_path, yaml_path = student_dir / "report.md", student_dir / "report.yaml"
    for path in (md_path, yaml_path):
        path.unlink(missing_ok=True)
    completed = _run(
        [
            f"{LAB06_DIR.name}/bridge.py",
            *("--verdicts", str(fixtures["paths"]["torchtpu"]), "--markdown", str(md_path), "--yaml", str(yaml_path)),
        ]
    )
    step.expect(completed.returncode == 0, f"bridge exited {completed.returncode}: {_tail(completed)}")
    if completed.returncode != 0 or not yaml_path.exists() or not md_path.exists():
        step.expect(yaml_path.exists() and md_path.exists(), f"bridge wrote no report under {student_dir}")
        return step
    report = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    statement = student["plans"]["26b_torchtpu"]["metadata"]["label_sources"]["statement"]
    step.expect(list(report) == ["rai_evaluation", "policy", "provenance"], f"top-level keys are {list(report)}")
    sections = report.get("rai_evaluation", {})
    step.expect(len(sections) == len(_schema.VIOLATION_STATES), f"{len(sections)} sections")
    flagged = {state for verdict in fixtures["right"].values() for state in _score_states(verdict)}
    for state in _schema.VIOLATION_STATES:
        section = sections.get(state.replace("_", " ").title(), {})
        want = "FAIL" if state in flagged else "PASS"
        step.expect(section.get("status") == want, f"{state}: status {section.get('status')}, expected {want}")
    step.expect(
        report.get("provenance", {}).get("label_sources") == statement,
        f"provenance names the labels as {report.get('provenance', {}).get('label_sources')!r}",
    )
    step.expect(_load_sibling("bridge").LABELS_SENTENCE in markdown, "the Markdown does not say it read no label")
    statuses = [section.get("status") for section in sections.values()]
    step.observed.append(
        f"{len(sections)} sections: {statuses.count('FAIL')} FAIL, {statuses.count('PASS')} PASS; "
        "the provenance block carries the label statement"
    )
    return step


def _score_states(verdict: dict[str, Any]) -> set[str]:
    return {violation["state"] for violation in verdict.get("violations", [])}


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
    """Run all eleven steps and return every result. Later steps still run after a failure."""
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

    # The student track, dry: review, build, plan, compare, parity, bridge.
    if tamper:
        set_step, eval_set = step_student_set(tamper, staging, out_dir)
    else:
        set_step, eval_set = (
            StepResult("student set", ["the tamper drafts did not load; see the staged load step"]),
            None,
        )
    results.append(set_step)
    plan_step, student = step_student_plan(eval_set, out_dir)
    results.append(plan_step)
    student_fixtures = write_student_fixture_runs(student, out_dir) if student else None
    results.append(step_three_readers(student, student_fixtures, out_dir))
    results.append(step_student_parity(student, student_fixtures, out_dir))
    results.append(step_student_bridge(student, student_fixtures, out_dir))
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
