"""
Unit tests for 05_trajectory_eval corpus tooling: corpus.py,
generate_candidates.py, adjudicate.py. No TPU or vLLM required.

The tests that matter most are the guardrails. The validity of every Table 3
number rests on one boundary: nothing enters the scored corpus without human
adjudication. These tests make that boundary mechanical:

  - the corpus loader fails loudly on any adjudicated=false record
  - promote_record refuses unadjudicated records
  - the candidate generator physically cannot write outside staging/
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_MODULE_DIR = Path(__file__).parent.parent / "05_trajectory_eval"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, str(_MODULE_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_schema = _load_module("schema")
_canon = _load_module("canonicalize")
_corpus = _load_module("corpus")
_gen = _load_module("generate_candidates")
_adj = _load_module("adjudicate")


def _small_batch():
    return _gen.generate_candidates(per_state=2, clean=4, seed=1234)


def _tmp_path(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


def _write_jsonl(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_schema.record_to_dict(r)) + "\n")
    return path


# ── The guardrail ──────────────────────────────────────────────────────────────


class TestCorpusGuardrail(unittest.TestCase):
    def test_unadjudicated_record_cannot_enter_scored_corpus(self):
        # An adjudicated=false record anywhere in the corpus file fails the
        # entire load. No skip, no warning-and-continue.
        candidate = _small_batch()[0]
        self.assertFalse(candidate.adjudicated)
        path = _write_jsonl(_tmp_path("trajectories.jsonl"), [candidate])
        with self.assertRaises(_corpus.CorpusIntegrityError):
            _corpus.load_corpus(path)

    def test_one_bad_record_poisons_the_whole_load(self):
        good = _small_batch()[0]
        good.labels = list(good.proposed_labels)
        good.adjudicated = True
        good.adjudication_note = "test 2026-08-03 accepted"
        bad = _small_batch()[1]
        path = _write_jsonl(_tmp_path("trajectories.jsonl"), [good, bad])
        with self.assertRaises(_corpus.CorpusIntegrityError):
            _corpus.load_corpus(path)

    def test_promote_refuses_unadjudicated_record(self):
        candidate = _small_batch()[0]
        corpus_path = _tmp_path("trajectories.jsonl")
        with self.assertRaises(_corpus.CorpusIntegrityError):
            _adj.promote_record(candidate, corpus_path)
        self.assertFalse(corpus_path.exists())

    def test_promote_appends_adjudicated_record(self):
        candidate = _small_batch()[0]
        candidate.labels = list(candidate.proposed_labels)
        candidate.adjudicated = True
        candidate.adjudication_note = "test 2026-08-03 accepted"
        corpus_path = _tmp_path("trajectories.jsonl")
        _adj.promote_record(candidate, corpus_path)
        loaded = _corpus.load_corpus(corpus_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, candidate.id)
        self.assertTrue(loaded[0].adjudicated)

    def test_generator_refuses_paths_outside_staging(self):
        records = _small_batch()
        for bad in ("sample_data/trajectories.jsonl", "results/candidates.jsonl", "candidates.jsonl"):
            with self.assertRaises(ValueError):
                _gen.write_candidates(records, _tmp_path("x").parent / bad)

    def test_generator_writes_inside_staging(self):
        records = _small_batch()[:3]
        out = _tmp_path("ignored").parent / "staging" / "candidates.jsonl"
        _gen.write_candidates(records, out)
        self.assertEqual(len(_corpus.load_staged(out)), 3)

    def test_duplicate_ids_rejected(self):
        record = _small_batch()[0]
        path = _write_jsonl(_tmp_path("staged.jsonl"), [record, record])
        with self.assertRaises(_corpus.CorpusIntegrityError):
            _corpus.load_staged(path)


# ── Generated candidates ───────────────────────────────────────────────────────


class TestGeneratedCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = _gen.generate_candidates(per_state=8, clean=20, seed=99)

    def test_generation_is_deterministic(self):
        again = _gen.generate_candidates(per_state=8, clean=20, seed=99)
        as_json = [json.dumps(_schema.record_to_dict(r), sort_keys=True) for r in self.records]
        again_json = [json.dumps(_schema.record_to_dict(r), sort_keys=True) for r in again]
        self.assertEqual(as_json, again_json)

    def test_all_candidates_validate_and_are_unadjudicated(self):
        for record in self.records:
            _schema.validate_record(record)
            self.assertFalse(record.adjudicated)
            self.assertEqual(record.labels, [])
            self.assertEqual(record.tier, "authored_clean")

    def test_every_state_has_proposed_positives(self):
        by_state = {state: 0 for state in _schema.VIOLATION_STATES}
        for record in self.records:
            for label in record.proposed_labels:
                by_state[label.state] += 1
        for state, count in by_state.items():
            self.assertGreaterEqual(count, 8, f"{state} under-generated")

    def test_clean_negatives_have_no_proposals(self):
        clean = [r for r in self.records if not r.proposed_labels]
        self.assertEqual(len(clean), 20)

    def test_bypass_exists_in_both_missing_gate_shapes(self):
        shapes = set()
        for record in self.records:
            for label in record.proposed_labels:
                if label.state != "authorization_bypass":
                    continue
                step = self._step_at(record, label.step_index)
                shapes.add(step.approval_state)
        self.assertIn("required_not_sought", shapes)
        self.assertIn("requested", shapes)

    def test_policy_variant_pairs_are_identical_except_approval(self):
        pv = {r.id[:-3]: r for r in self.records if r.id.endswith("-pv")}
        pw = {r.id[:-3]: r for r in self.records if r.id.endswith("-pw")}
        self.assertTrue(pv, "no policy_visible records generated")
        self.assertEqual(set(pv), set(pw), "pv/pw pairs do not line up")
        for base, visible in pv.items():
            withheld = pw[base]
            self.assertEqual(visible.variant, "policy_visible")
            self.assertEqual(withheld.variant, "policy_withheld")
            self.assertEqual(visible.proposed_labels, withheld.proposed_labels)
            at = visible.proposed_labels[0].step_index
            self.assertEqual(self._step_at(visible, at).approval_state, "required_not_sought")
            self.assertEqual(self._step_at(withheld, at).approval_state, "not_required")
            # Everything except the policy field on the violation step matches.
            for sv, sw in zip(visible.steps, withheld.steps):
                if sv.index == at:
                    continue
                self.assertEqual(sv, sw)

    def test_step_count_distribution_is_wide_and_crosses_the_cap(self):
        counts = [len(r.steps) + len(r.overflow_steps) for r in self.records]
        self.assertLess(min(counts), 10)
        self.assertGreater(max(counts), _schema.DEFAULT_MAX_STEPS)
        truncated = [r for r in self.records if r.truncation is not None]
        self.assertTrue(truncated, "no records exercise the truncation path")

    def test_every_state_has_positives_in_both_regions(self):
        # Reduced-region positives exist for every state on purpose, even
        # the states whose evidence reduction drops: they populate the
        # instrumentation-dependency table. Full-fidelity positives carry
        # the headline results.
        by_state_region = {state: set() for state in _schema.VIOLATION_STATES}
        for record in self.records:
            for label in record.proposed_labels:
                by_state_region[label.state].add(label.region)
        for state, regions in by_state_region.items():
            self.assertIn("full_fidelity", regions, f"{state} has no full-fidelity positives")
            self.assertIn("reduced_region", regions, f"{state} has no reduced-region positives")

    def test_region_tags_match_truncation_geometry(self):
        for record in self.records:
            reduced_ids = {s.index for s in record.overflow_steps}
            for label in record.proposed_labels:
                expected = "reduced_region" if label.step_index in reduced_ids else "full_fidelity"
                self.assertEqual(label.region, expected, f"{record.id} {label.state}@{label.step_index}")

    def _step_at(self, record, index):
        for s in list(record.steps) + list(record.overflow_steps):
            if s.index == index:
                return s
        raise AssertionError(f"step {index} missing on {record.id}")


# ── Balance reporting and adjudication helpers ─────────────────────────────────


class TestBalanceAndAdjudication(unittest.TestCase):
    def test_class_balance_counts_and_flags(self):
        records = _gen.generate_candidates(per_state=2, clean=3, seed=7)
        balance = _corpus.class_balance(records, label_source="proposed")
        self.assertEqual(balance["clean_negatives"], 3)
        for state in _schema.VIOLATION_STATES:
            self.assertGreaterEqual(balance["per_state"][state]["total"], 2)
        # Way under the 25-per-state target, so everything is flagged.
        self.assertEqual(list(balance["under_filled"]), list(_schema.VIOLATION_STATES))
        self.assertEqual(balance["clean_shortfall"], _corpus.TARGET_CLEAN_NEGATIVES - 3)

    def test_class_balance_on_empty_corpus(self):
        balance = _corpus.class_balance([], label_source="labels")
        self.assertEqual(balance["total_records"], 0)
        self.assertEqual(balance["clean_negatives"], 0)
        self.assertEqual(len(balance["under_filled"]), 6)

    def test_class_balance_rejects_unknown_label_source(self):
        with self.assertRaises(ValueError):
            _corpus.class_balance([], label_source="vibes")

    def test_missing_corpus_loads_empty(self):
        self.assertEqual(_corpus.load_corpus(_tmp_path("missing.jsonl")), [])

    def test_parse_label_edit_valid(self):
        labels = _adj.parse_label_edit("scope_violation:3, misreported_state:7", {3, 7})
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].state, "scope_violation")
        self.assertEqual(labels[1].step_index, 7)

    def test_parse_label_edit_rejects_unknown_state(self):
        with self.assertRaises(ValueError):
            _adj.parse_label_edit("vibes_violation:3", {3})

    def test_parse_label_edit_rejects_missing_step(self):
        with self.assertRaises(ValueError):
            _adj.parse_label_edit("scope_violation:99", {3})

    def test_parse_label_edit_rejects_empty(self):
        with self.assertRaises(ValueError):
            _adj.parse_label_edit("", {3})

    def test_stratified_sample_is_deterministic_clean_only_and_stratified(self):
        records = _gen.generate_candidates(per_state=4, clean=16, seed=44)
        sampled = _adj.stratified_sample(records, rate=0.25, seed=7)
        again = _adj.stratified_sample(records, rate=0.25, seed=7)
        self.assertEqual([r.id for r in sampled], [r.id for r in again])
        self.assertTrue(all(not r.proposed_labels for r in sampled))
        # Every clean sub-scenario stratum contributes at least one record.
        strata = {r.provenance for r in records if not r.proposed_labels}
        self.assertSetEqual({r.provenance for r in sampled}, strata)
        with self.assertRaises(ValueError):
            _adj.stratified_sample(records, rate=0.0, seed=7)

    def test_accept_remainder_records_protocol_in_every_note(self):
        records = _gen.generate_candidates(per_state=2, clean=8, seed=45)
        remainder = [r for r in records if not r.proposed_labels][:5]
        corpus_path = _tmp_path("trajectories.jsonl")
        n = _adj.accept_remainder_on_construction(
            remainder, corpus_path, adjudicator="test", rate=0.2, sampled_n=3, disagreements=0
        )
        self.assertEqual(n, 5)
        loaded = _corpus.load_corpus(corpus_path)
        self.assertEqual(len(loaded), 5)
        for record in loaded:
            self.assertTrue(record.adjudicated)
            self.assertEqual(record.labels, [])
            self.assertIn("construction guarantee", record.adjudication_note)
            self.assertIn("spot-check rate 20%", record.adjudication_note)
            self.assertIn("disagreements 0", record.adjudication_note)


if __name__ == "__main__":
    unittest.main()
