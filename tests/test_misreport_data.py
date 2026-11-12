"""
Unit tests for module 06's committed data files.

The review files under 06_misreported_state_torchtpu/data/staging/ are cut
from sample_data/staging/candidates_tier2_batch1.jsonl by id. These tests
pin two things: the committed files are byte for byte what
build_review_files.py produces, and every staged file in 06 loads through
05's own loader. A third class pins how the eight tamper records describe
their own origin: drafted with AI assistance, never hand-authored. No TPU,
no vllm.
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


class TestTamperProvenance(unittest.TestCase):
    """The eight were drafted with AI assistance from the public report. Nothing may say otherwise."""

    @classmethod
    def setUpClass(cls):
        cls.tamper = _corpus.load_staged(_STAGING / "candidates_tamper_batch1.jsonl")
        cls.readme = (_LAB06 / "README.md").read_text(encoding="utf-8")

    def test_provenance_text_says_ai_assisted_draft_modelled_on_the_report(self):
        for record in self.tamper:
            self.assertTrue(
                record.provenance.startswith("ai-assisted-draft/modelled-on-oct20-misalignment-report/"), record.id
            )
            self.assertLessEqual(len(record.provenance), _schema.MAX_SHORT_CHARS, record.id)
            self.assertEqual(record.source, "tier3_ai_assisted_draft", record.id)

    def test_no_free_text_field_claims_the_records_were_authored(self):
        for record in self.tamper:
            for field in (record.provenance, record.source, record.adjudication_note):
                self.assertNotIn("authored", field, record.id)
                self.assertNotIn("hand", field.split("/")[0], record.id)

    def test_tier_value_is_kept_because_05s_vocabulary_has_no_closer_word(self):
        self.assertEqual({record.tier for record in self.tamper}, {"authored_subtle"})
        self.assertEqual(_schema.TIERS, ("real", "authored_clean", "authored_subtle"))

    def test_readme_states_how_the_records_were_made(self):
        self.assertIn("drafted with AI assistance from the public report", self.readme)
        self.assertIn("reviewed by a human during adjudication", self.readme)
        self.assertIn("nobody wrote them by hand", self.readme)

    def test_readme_never_asks_for_or_claims_hand_authoring(self):
        lowered = self.readme.lower()
        for phrase in ("hand-authored tamper", "rewrite their", "wrote the tamper", "authored `checkout-api`"):
            self.assertNotIn(phrase, lowered)
        # The one place hand-authoring is mentioned is the warning that 05's tier definition does not apply.
        mentions = [line for line in self.readme.splitlines() if "hand-authored" in line]
        self.assertEqual(len(mentions), 1)
        self.assertIn("these eight are not", mentions[0])


if __name__ == "__main__":
    unittest.main()
