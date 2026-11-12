"""
Unit tests for 06_misreported_state_torchtpu/rehearsal.py and its make target.

The rehearsal is itself a check, so these tests pin two things: it passes on
the committed files, and it fails, naming the step, when a staged file drifts.
It runs the 05 and 06 scripts as subprocesses the way the README does,
including the student track's steps on fixtures. No TPU, no vllm.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
_LAB06 = _ROOT / "06_misreported_state_torchtpu"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# lab05 first: the 06 scripts import it as a sibling.
_lab05 = _load("lab05", _LAB06 / "lab05.py")
_rehearsal = _load("misreport_rehearsal", _LAB06 / "rehearsal.py")
_schema = _lab05.load("schema")
_corpus = _lab05.load("corpus")
_labels = _lab05.sibling("labels")

TOOL_STEPS = ["review files", "staged load", "rules baseline", "plan diff", "parity", "bridge"]
STUDENT_STEPS = ["student set", "student plan", "three readers", "student parity", "student bridge"]
STEP_NAMES = TOOL_STEPS + STUDENT_STEPS


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp())


class TestRehearsalPasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_dir = _tmp_dir()
        cls.results = _rehearsal.rehearse(out_dir=cls.out_dir)

    def test_eleven_steps_in_order_all_ok(self):
        self.assertEqual([step.name for step in self.results], STEP_NAMES)
        self.assertEqual(len(self.results), 11)
        self.assertEqual({step.name: step.problems for step in self.results if not step.ok}, {})

    def test_every_step_reports_what_it_observed(self):
        for step in self.results:
            self.assertTrue(step.observed, f"{step.name} recorded no observation")

    def test_plans_from_05_and_06_are_identical(self):
        plan05 = json.loads((self.out_dir / "plan05.json").read_text(encoding="utf-8"))
        plan06 = json.loads((self.out_dir / "plan06.json").read_text(encoding="utf-8"))
        self.assertEqual(plan05["plan"], plan06["plan"])
        self.assertEqual(sum(entry["prompts"] for entry in plan06["plan"]), 12)

    def test_fixtures_are_marked_and_never_look_like_results(self):
        for backend in ("jax", "torchtpu"):
            doc = json.loads((self.out_dir / f"fixture_verdicts_{backend}.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["metadata"]["evaluator"], _rehearsal.FIXTURE_EVALUATOR)
            self.assertEqual(doc["metadata"]["backend"], backend)
            self.assertIn("never publish", doc["metadata"]["note"])
            self.assertEqual(doc["metadata"]["vllm_distribution"]["name"], "unchecked (dry run)")

    def test_stand_in_band_reports_are_marked_staged(self):
        for name in ("stand_in_bands_tier2.json", "stand_in_bands_tamper.json"):
            report = json.loads((self.out_dir / name).read_text(encoding="utf-8"))
            self.assertTrue(report["metadata"]["staged"])
            self.assertIn("stand-in", report["metadata"]["tokenizer"])

    def test_parity_finds_exactly_the_planted_disagreement(self):
        report = json.loads((self.out_dir / "fixture_parity.json").read_text(encoding="utf-8"))
        self.assertEqual(report["overall"]["decisions"], 8 * len(_schema.VIOLATION_STATES))
        self.assertEqual(report["overall"]["agree"], report["overall"]["decisions"] - 1)
        disagreements = [ids for entry in report["agreement"].values() if (ids := entry["disagree_ids"])]
        self.assertEqual(len(disagreements), 1)

    def test_refusals_wrote_nothing(self):
        self.assertFalse((self.out_dir / "parity_should_not_exist.json").exists())
        self.assertFalse((self.out_dir / "bad_should_not_exist.yaml").exists())
        self.assertFalse((self.out_dir / "student" / "parity_should_not_exist.json").exists())
        self.assertFalse((self.out_dir / "student" / "readers_should_not_exist.json").exists())


class TestStudentTrackDryRun(unittest.TestCase):
    """Steps 7 to 11: the student track on fixtures, with the same commands the README gives."""

    @classmethod
    def setUpClass(cls):
        cls.out_dir = _tmp_dir()
        cls.results = {step.name: step for step in _rehearsal.rehearse(out_dir=cls.out_dir)}
        cls.student = cls.out_dir / "student"

    def _json(self, name: str) -> dict:
        return json.loads((self.student / name).read_text(encoding="utf-8"))

    def test_student_steps_pass_and_say_what_they_saw(self):
        for name in STUDENT_STEPS:
            self.assertTrue(self.results[name].ok, self.results[name].problems)
            self.assertTrue(self.results[name].observed, name)

    def test_adjudication_ran_through_adjudicate_py_under_the_fixture_name(self):
        decided = _corpus.load_corpus(self.student / "tamper_labels.jsonl")
        self.assertEqual(len(decided), 8)
        for record in decided:
            self.assertTrue(record.adjudication_note.startswith(f"{_rehearsal.FIXTURE_ADJUDICATOR} "), record.id)
            self.assertEqual(record.labels, record.proposed_labels, record.id)
        self.assertEqual(sum(1 for record in decided if not record.labels), 3)

    def test_eval_set_is_22_records_and_names_its_labels(self):
        records = _corpus.load_staged(self.student / "eval_set.jsonl")
        self.assertEqual(len(records), 22)
        summary = _labels.summarize(records, staged=True)
        self.assertEqual(
            summary["statement"],
            "tier authored_clean, 14 records: template-construction labels, not adjudicated; "
            "tier authored_subtle, 8 records: student-adjudicated (by rehearsal-fixture)",
        )

    def test_fixture_adjudication_never_lands_where_a_students_review_goes(self):
        self.assertEqual(self.student, self.out_dir / _rehearsal.STUDENT_SUBDIR)
        self.assertNotEqual(self.student.resolve(), (_ROOT / "results" / "student").resolve())
        self.assertEqual(_rehearsal.DEFAULT_OUT_DIR, _ROOT / "results" / "rehearsal")

    def test_both_backends_and_the_fallback_model_plan_132_prompts(self):
        for name in ("plan_26b_jax.json", "plan_26b_torchtpu.json", "plan_e4b_jax.json", "plan_e4b_torchtpu.json"):
            plan = self._json(name)
            self.assertFalse(plan["executed"], name)
            self.assertEqual(sum(entry["prompts"] for entry in plan["plan"]), 132, name)
        self.assertEqual(self._json("plan_e4b_jax.json")["metadata"]["model"], _rehearsal.FALLBACK_MODEL)
        self.assertEqual(self._json("plan_26b_jax.json")["metadata"]["model"], _rehearsal.MODEL)

    def test_student_fixtures_are_marked_and_never_look_like_results(self):
        for name in ("jax", "torchtpu", "e4b_jax", "e4b_torchtpu"):
            meta = self._json(f"fixture_verdicts_{name}.json")["metadata"]
            self.assertEqual(meta["evaluator"], _rehearsal.FIXTURE_EVALUATOR)
            self.assertIn("never publish", meta["note"])
            self.assertEqual(meta["vllm_distribution"]["name"], "unchecked (dry run)")

    def test_three_readers_file_has_a_row_per_record_and_no_rate(self):
        readers = self._json("readers.json")
        self.assertEqual(len(readers["cases"]), 22)
        self.assertEqual(readers["counts"]["authored_subtle"]["records"], 8)
        self.assertEqual(readers["counts"]["authored_subtle"]["rules_flagged_any"], 1)
        self.assertEqual(readers["counts"]["authored_subtle"]["judge_same_states_as_labels"], 7)
        self.assertEqual(_rehearsal._rate_keys(readers), [])

    def test_student_parity_finds_the_one_planted_disagreement_and_gives_counts(self):
        parity = self._json("parity.json")
        self.assertEqual(parity["overall"]["decisions"], 22 * len(_schema.VIOLATION_STATES))
        self.assertEqual(parity["overall"]["agree"], parity["overall"]["decisions"] - 1)
        self.assertEqual(parity["agreement"]["scope_violation"]["disagree_ids"], ["traj-t3-tmp-001"])
        for tier in ("authored_subtle", "authored_clean"):
            self.assertTrue(parity["labels"]["tier_notes"][tier]["counts_only"], tier)
        self.assertTrue((self.student / "parity_e4b.json").exists())

    def test_student_report_carries_the_label_statement(self):
        report = yaml.safe_load((self.student / "report.yaml").read_text(encoding="utf-8"))
        self.assertIn("student-adjudicated (by rehearsal-fixture)", report["provenance"]["label_sources"])
        self.assertIn("template-construction labels, not adjudicated", report["provenance"]["label_sources"])

    def test_rate_key_scan_finds_a_rate_when_there_is_one(self):
        self.assertEqual(
            _rehearsal._rate_keys({"a": [{"recall": 1.0}], "agreement_rate": 0.5}), ["agreement_rate", "recall"]
        )


class TestRehearsalFailsOnMismatch(unittest.TestCase):
    def _drifted_staging(self) -> Path:
        staging = _tmp_dir() / "staging"
        shutil.copytree(_LAB06 / "data" / "staging", staging)
        cut60 = staging / _rehearsal.CUT60_FILENAME
        lines = cut60.read_text(encoding="utf-8").splitlines(keepends=True)
        cut60.write_text("".join(lines[:-1]), encoding="utf-8")
        return staging

    def test_drifted_review_file_fails_the_steps_that_read_it(self):
        results = _rehearsal.rehearse(out_dir=_tmp_dir(), staging=self._drifted_staging())
        failed = {step.name: step.problems for step in results if not step.ok}
        self.assertEqual(set(failed), {"review files", "staged load"})
        self.assertIn("differs from a rebuild", failed["review files"][0])
        self.assertIn("expected 60 records, loaded 59", failed["staged load"][0])

    def test_drifted_student_sample_fails_the_student_set_step(self):
        staging = _tmp_dir() / "staging"
        shutil.copytree(_LAB06 / "data" / "staging", staging)
        sample = staging / _rehearsal.STUDENT14_FILENAME
        lines = sample.read_text(encoding="utf-8").splitlines(keepends=True)
        sample.write_text("".join(lines[:-1]), encoding="utf-8")
        results = _rehearsal.rehearse(out_dir=_tmp_dir(), staging=staging)
        failed = {step.name: step.problems for step in results if not step.ok}
        self.assertIn("student set", failed)
        self.assertIn("eval set has 21 records", failed["student set"][0])

    def test_main_exits_1_on_mismatch(self):
        args = argparse.Namespace(
            out_dir=str(_tmp_dir()), staging=str(self._drifted_staging()), source=str(_rehearsal.DEFAULT_SOURCE)
        )
        with self.assertRaises(SystemExit) as ctx:
            _rehearsal.main(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_changed_rules_expectation_is_a_mismatch(self):
        tamper = _lab05.load("corpus").load_staged(_LAB06 / "data" / "staging" / _rehearsal.TAMPER_FILENAME)
        step = _rehearsal.step_rules_baseline(tamper[1:], _tmp_dir())
        self.assertFalse(step.ok)
        self.assertIn("rules flagged []", step.problems[0])


class TestMakeTarget(unittest.TestCase):
    def test_makefile_runs_the_rehearsal_script(self):
        makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("\nmisreport-rehearsal:", makefile)
        recipe = makefile.split("\nmisreport-rehearsal:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("python $(MISREPORT_DIR)/rehearsal.py", recipe)
        phony = makefile.split(".PHONY:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("misreport-rehearsal", phony)


if __name__ == "__main__":
    unittest.main()
