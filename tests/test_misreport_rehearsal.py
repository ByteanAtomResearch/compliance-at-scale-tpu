"""
Unit tests for 06_misreported_state_torchtpu/rehearsal.py and its make target.

The rehearsal is itself a check, so these tests pin two things: it passes on
the committed files, and it fails, naming the step, when a staged file drifts.
It runs the 05 and 06 scripts as subprocesses the way the README does. No
TPU, no vllm.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

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

STEP_NAMES = ["review files", "staged load", "rules baseline", "plan diff", "parity", "bridge"]


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp())


class TestRehearsalPasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_dir = _tmp_dir()
        cls.results = _rehearsal.rehearse(out_dir=cls.out_dir)

    def test_six_steps_in_lab_order_all_ok(self):
        self.assertEqual([step.name for step in self.results], STEP_NAMES)
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
