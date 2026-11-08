"""
Unit tests for 06_misreported_state_torchtpu/bridge.py.

The bridge reproduces 04's rai_evaluation shape for 05's six states and
keeps operator policy out of the findings. These tests pin the section
order, the PASS/FAIL rule, the separate policy and provenance blocks, and
the refusal on a verdict 05's schema rejects. No TPU, no vllm.
"""

import argparse
import importlib.util
import json
import re
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


_lab05 = _load("lab05", _LAB06 / "lab05.py")
_bridge = _load("misreport_bridge", _LAB06 / "bridge.py")
_schema = _lab05.load("schema")
_gen = _lab05.load("generate_candidates")
_rules = _lab05.load("rules_baseline")

POLICY_SOURCE = "operator policy lookup (DEFAULT_ENFORCEMENT_POLICY), not model output"
ENFORCEMENT_WORDS = ("recommended_enforcement", "escalate", "block", "review")


def _tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


def _verdict(*states: str, enforcement: str = "none", step: int = 0, confidence: float = 0.5) -> dict:
    return {
        "violations": [
            {"state": state, "failed_step_index": step, "evidence": "fixture", "confidence": confidence}
            for state in states
        ],
        "policy": {"recommended_enforcement": enforcement, "source": POLICY_SOURCE},
    }


def _verdicts() -> dict:
    return {
        "traj-c": _verdict(),
        "traj-b": _verdict("authorization_bypass", enforcement="block"),
        "traj-a": _verdict("authorization_bypass", "misreported_state", enforcement="escalate"),
    }


def _doc05(verdicts: dict) -> dict:
    return {
        "metadata": {
            "model": "google/gemma-4-E4B-it",
            "prompt_version": "part2-judge-v1",
            "prompt_hash": "p" * 64,
            "call_mode": "six_call",
            "parse_errors": 0,
            "model_revision": "unpinned (record before publishing)",
            "container_image_digest": "unrecorded (set before publishing)",
        },
        "bands": [],
        "verdicts": verdicts,
    }


def _doc06(verdicts: dict) -> dict:
    doc = _doc05(verdicts)
    doc["metadata"].update(
        {"backend": "torchtpu", "dtype": "bfloat16", "git": {"sha": "abc", "branch": "b", "dirty": False}}
    )
    return doc


def _run_bridge(doc: dict, markdown: bool = True, yaml_out: bool = True):
    verdict_path = _tmp("verdicts.json")
    verdict_path.write_text(json.dumps(doc), encoding="utf-8")
    md_path = _tmp("report.md") if markdown else None
    yaml_path = _tmp("report.yaml") if yaml_out else None
    _bridge.main(
        argparse.Namespace(
            verdicts=str(verdict_path),
            markdown=str(md_path) if md_path else None,
            yaml=str(yaml_path) if yaml_path else None,
        )
    )
    md_text = md_path.read_text(encoding="utf-8") if md_path else None
    yaml_doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path else None
    return md_text, yaml_doc


def _expect_refusal(test: unittest.TestCase, doc: dict, needle: str | None = None) -> None:
    verdict_path = _tmp("verdicts.json")
    verdict_path.write_text(json.dumps(doc), encoding="utf-8")
    md_path, yaml_path = _tmp("report.md"), _tmp("report.yaml")
    with test.assertRaises(SystemExit) as ctx:
        _bridge.main(argparse.Namespace(verdicts=str(verdict_path), markdown=str(md_path), yaml=str(yaml_path)))
    test.assertNotEqual(ctx.exception.code, 0)
    if needle:
        test.assertIn(needle, str(ctx.exception))
    test.assertFalse(md_path.exists())
    test.assertFalse(yaml_path.exists())


# ── Validation ────────────────────────────────────────────────────────────────


class TestValidation(unittest.TestCase):
    def test_unknown_state_exits_nonzero_and_writes_nothing(self):
        verdicts = _verdicts()
        verdicts["traj-c"]["violations"].append(
            {"state": "vibes", "failed_step_index": 0, "evidence": "x", "confidence": 0.5}
        )
        _expect_refusal(self, _doc06(verdicts), needle="vibes")

    def test_unknown_enforcement_exits_nonzero(self):
        verdicts = _verdicts()
        verdicts["traj-b"]["policy"]["recommended_enforcement"] = "shrug"
        _expect_refusal(self, _doc06(verdicts), needle="shrug")

    def test_malformed_violation_keys_exit_nonzero(self):
        verdicts = _verdicts()
        del verdicts["traj-b"]["violations"][0]["confidence"]
        _expect_refusal(self, _doc06(verdicts), needle="traj-b")

    def test_confidence_out_of_range_exits_nonzero(self):
        _expect_refusal(self, _doc06({"traj-x": _verdict("scope_violation", confidence=1.5)}), needle="confidence")

    def test_no_output_flag_is_an_error(self):
        verdict_path = _tmp("verdicts.json")
        verdict_path.write_text(json.dumps(_doc06(_verdicts())), encoding="utf-8")
        with self.assertRaises(SystemExit):
            _bridge.main(argparse.Namespace(verdicts=str(verdict_path), markdown=None, yaml=None))

    def test_file_without_verdicts_is_refused(self):
        verdict_path = _tmp("plan.json")
        verdict_path.write_text(json.dumps({"metadata": {}, "plan": []}), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            _bridge.main(argparse.Namespace(verdicts=str(verdict_path), markdown=None, yaml=str(_tmp("r.yaml"))))
        self.assertIn("verdicts", str(ctx.exception))


# ── Markdown ──────────────────────────────────────────────────────────────────


class TestMarkdown(unittest.TestCase):
    def setUp(self):
        self.md, _ = _run_bridge(_doc06(_verdicts()))

    def test_six_sections_in_violation_states_order(self):
        headings = re.findall(r"^## (.+)$", self.md, re.MULTILINE)
        self.assertEqual(headings[:6], [_bridge.title_for(state) for state in _schema.VIOLATION_STATES])
        self.assertEqual(headings[6:], ["Policy", "Provenance"])

    def test_pass_and_fail_lines_match_04_shape(self):
        self.assertIn("- [x] All records passed (3 clean, 0 flagged)", self.md)
        self.assertIn("- [ ] Issues found: 2 flagged, 1 clean", self.md)
        self.assertIn("- [ ] Issues found: 1 flagged, 2 clean", self.md)
        self.assertIn("  - Flagged: traj-a", self.md)

    def test_flagged_ids_sorted(self):
        self.assertIn("  - Flagged: traj-a, traj-b", self.md)

    def test_policy_block_after_sections_never_inside(self):
        policy_at = self.md.index("## Policy")
        last_state_at = self.md.index(f"## {_bridge.title_for(_schema.VIOLATION_STATES[-1])}")
        self.assertGreater(policy_at, last_state_at)
        sections = self.md[self.md.index("## Scope Violation") : policy_at]
        for word in ENFORCEMENT_WORDS:
            self.assertNotIn(word, sections)
        self.assertIn("- traj-b: block", self.md[policy_at:])
        self.assertIn("- traj-a: escalate", self.md[policy_at:])
        self.assertIn(_bridge.POLICY_SENTENCE, self.md[policy_at:])

    def test_provenance_block_lists_required_keys(self):
        provenance = self.md[self.md.index("## Provenance") :]
        for key in (*_bridge.PROVENANCE_FIELDS, "git_sha"):
            self.assertIn(f"- {key}: ", provenance)
        self.assertIn("- git_sha: abc", provenance)

    def test_header_matches_04(self):
        self.assertTrue(self.md.startswith("# RAI Compliance Evaluation Report\n"))
        self.assertIn("**Model:** google/gemma-4-E4B-it", self.md)
        self.assertIn("**Records evaluated:** 3", self.md)
        self.assertIn("**Backend:** torchtpu", self.md)


# ── YAML ──────────────────────────────────────────────────────────────────────


class TestYaml(unittest.TestCase):
    def setUp(self):
        _, self.doc = _run_bridge(_doc06(_verdicts()))

    def test_rai_evaluation_has_six_titles_in_order(self):
        self.assertEqual(list(self.doc["rai_evaluation"]), [_bridge.title_for(s) for s in _schema.VIOLATION_STATES])

    def test_status_counts_and_records(self):
        self.assertEqual(
            self.doc["rai_evaluation"]["Authorization Bypass"],
            {"status": "FAIL", "flagged_count": 2, "flagged_records": ["traj-a", "traj-b"]},
        )
        self.assertEqual(
            self.doc["rai_evaluation"]["Misreported State"],
            {"status": "FAIL", "flagged_count": 1, "flagged_records": ["traj-a"]},
        )
        self.assertEqual(
            self.doc["rai_evaluation"]["Scope Violation"], {"status": "PASS", "flagged_count": 0, "flagged_records": []}
        )

    def test_policy_and_provenance_are_top_level_siblings(self):
        self.assertEqual(set(self.doc), {"rai_evaluation", "policy", "provenance"})
        for section in self.doc["rai_evaluation"].values():
            self.assertEqual(set(section), {"status", "flagged_count", "flagged_records"})

    def test_policy_source_comes_from_verdict(self):
        self.assertEqual(self.doc["policy"]["source"], POLICY_SOURCE)
        self.assertEqual(
            self.doc["policy"]["recommended_enforcement"], {"traj-c": "none", "traj-b": "block", "traj-a": "escalate"}
        )


# ── Inputs ────────────────────────────────────────────────────────────────────


class TestInputs(unittest.TestCase):
    def test_05_shaped_file_marks_backend_unrecorded(self):
        md, doc = _run_bridge(_doc05(_verdicts()))
        self.assertEqual(doc["provenance"]["backend"], _bridge.UNRECORDED)
        self.assertEqual(doc["provenance"]["git_sha"], _bridge.UNRECORDED)
        self.assertEqual(doc["provenance"]["model"], "google/gemma-4-E4B-it")
        self.assertIn("**Backend:** unrecorded", md)

    def test_06_shaped_file_carries_backend_and_git_sha(self):
        _, doc = _run_bridge(_doc06(_verdicts()))
        self.assertEqual(doc["provenance"]["backend"], "torchtpu")
        self.assertEqual(doc["provenance"]["dtype"], "bfloat16")
        self.assertEqual(doc["provenance"]["git_sha"], "abc")
        self.assertEqual(doc["provenance"]["parse_errors"], 0)

    def test_rules_baseline_file_works(self):
        records = _gen.generate_candidates(per_state=1, clean=2, seed=5)
        rules_doc = {"metadata": {"evaluator": "rules_baseline"}, "verdicts": _rules.evaluate_records(records)}
        _, doc = _run_bridge(rules_doc)
        self.assertEqual(doc["provenance"]["model"], _bridge.UNRECORDED)
        self.assertEqual(list(doc["rai_evaluation"]), [_bridge.title_for(s) for s in _schema.VIOLATION_STATES])
        self.assertEqual(set(doc["policy"]["recommended_enforcement"]), {r.id for r in records})

    def test_yaml_only_invocation_allowed(self):
        md, doc = _run_bridge(_doc06(_verdicts()), markdown=False, yaml_out=True)
        self.assertIsNone(md)
        self.assertIn("rai_evaluation", doc)


if __name__ == "__main__":
    unittest.main()
