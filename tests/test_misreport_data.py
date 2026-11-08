"""
Unit tests for module 06's committed data files.

The review files under 06_misreported_state_torchtpu/data/staging/ are cut
from sample_data/staging/candidates_tier2_batch1.jsonl by id. These tests
pin two things: the committed files are byte for byte what
build_review_files.py produces, and every staged file in 06 loads through
05's own loader. No TPU, no vllm.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_LAB06 = _ROOT / "06_misreported_state_torchtpu"
_STAGING = _LAB06 / "data" / "staging"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# lab05 first: the 06 scripts import it as a sibling.
_lab05 = _load("lab05", _LAB06 / "lab05.py")
_build = _load("misreport_build_review_files", _LAB06 / "build_review_files.py")
_corpus = _lab05.load("corpus")
_schema = _lab05.load("schema")


def _source_ids() -> set[str]:
    with open(_build.DEFAULT_SOURCE, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


class TestIdLists(unittest.TestCase):
    def test_sizes_and_disjointness(self):
        self.assertEqual(len(_build.CUT60_IDS), 60)
        self.assertEqual(len(set(_build.CUT60_IDS)), 60)
        self.assertEqual(len(_build.TOPUP14_IDS), 14)
        self.assertEqual(len(set(_build.TOPUP14_IDS)), 14)
        self.assertFalse(set(_build.CUT60_IDS) & set(_build.TOPUP14_IDS))

    def test_every_id_exists_in_the_staged_batch(self):
        present = _source_ids()
        self.assertFalse(set(_build.CUT60_IDS) - present)
        self.assertFalse(set(_build.TOPUP14_IDS) - present)

    def test_select_lines_names_missing_ids(self):
        with self.assertRaises(SystemExit) as ctx:
            _build.select_lines(_build.DEFAULT_SOURCE, ("traj-t2-nope-999",))
        self.assertIn("traj-t2-nope-999", str(ctx.exception))


class TestCommittedFiles(unittest.TestCase):
    def test_review_files_rebuild_byte_identical(self):
        self.assertEqual(_build.check(_build.DEFAULT_SOURCE, _STAGING), [])

    def test_lines_are_verbatim_source_lines(self):
        with open(_build.DEFAULT_SOURCE, encoding="utf-8", newline="") as f:
            source_lines = {json.loads(line)["id"]: line for line in f if line.strip()}
        for filename in (_build.CUT60_FILENAME, _build.TOPUP14_FILENAME):
            with open(_STAGING / filename, encoding="utf-8", newline="") as f:
                for line in f:
                    self.assertEqual(line, source_lines[json.loads(line)["id"]])

    def test_review_files_load_as_staged_with_expected_counts(self):
        cut60 = _corpus.load_staged(_STAGING / _build.CUT60_FILENAME)
        topup = _corpus.load_staged(_STAGING / _build.TOPUP14_FILENAME)
        tamper = _corpus.load_staged(_STAGING / "candidates_tamper_batch1.jsonl")
        self.assertEqual(len(cut60), 60)
        self.assertEqual(len(topup), 14)
        self.assertEqual(len(tamper), 8)
        # 10 proposed positives per state in the cut, as the spec's draw says.
        per_state = {state: 0 for state in _schema.VIOLATION_STATES}
        for record in cut60:
            for label in record.proposed_labels:
                per_state[label.state] += 1
        self.assertEqual(set(per_state.values()), {10})
        # Every top-up record is an authorization_bypass policy pair.
        self.assertTrue(all(r.variant in ("policy_visible", "policy_withheld") for r in topup))
        # Nothing staged is adjudicated: adjudicate.py is still the only path in.
        self.assertFalse(any(r.adjudicated for r in cut60 + topup + tamper))


if __name__ == "__main__":
    unittest.main()
