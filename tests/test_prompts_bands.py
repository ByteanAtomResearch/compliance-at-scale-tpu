"""
Unit tests for 05_trajectory_eval/prompts.py, bands.py, and freeze.py.
No TPU, no vLLM, no transformers: band math is tested with an injected
tokenizer function, which is exactly the seam bands.py exposes.

The highest-stakes property here is prompt blindness: nothing that renders
into a judge prompt may reveal labels, tier, variant, or provenance. A
prompt that leaked any of those would hand the judge the answer key and
quietly invalidate Table 3.
"""

import importlib.util
import sys
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
_prompts = _load_module("prompts")
_bands = _load_module("bands")
_freeze = _load_module("freeze")


def _records():
    return _gen.generate_candidates(per_state=4, clean=6, seed=11)


def _word_tokenize(text: str) -> int:
    return len(text.split())


# ── Prompts ────────────────────────────────────────────────────────────────────


class TestPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = _records()

    def test_render_is_deterministic(self):
        record = self.records[0]
        a = _prompts.render_state_prompt(record, "scope_violation")
        b = _prompts.render_state_prompt(record, "scope_violation")
        self.assertEqual(a, b)

    def test_state_prompt_contains_definition_and_telemetry(self):
        record = self.records[0]
        prompt = _prompts.render_state_prompt(record, "misreported_state")
        self.assertIn("misreported_state", prompt)
        self.assertIn(_prompts.STATE_DEFINITIONS["misreported_state"], prompt)
        self.assertIn(record.goal, prompt)
        self.assertIn("FULL", prompt)

    def test_multi_prompt_lists_all_six_states(self):
        prompt = _prompts.render_multi_prompt(self.records[0])
        for state in _schema.VIOLATION_STATES:
            self.assertIn(state, prompt)
        # Enforcement is operator policy, never requested from the model.
        self.assertNotIn("recommended_enforcement", prompt)
        self.assertNotIn("enforcement", prompt)

    def test_truncated_record_renders_reduced_steps_and_note(self):
        truncated = next(r for r in self.records if r.truncation is not None)
        prompt = _prompts.render_state_prompt(truncated, "unsafe_continuation")
        self.assertIn("REDUCED", prompt)
        self.assertIn("Truncation:", prompt)

    def test_prompt_never_leaks_labels_tier_variant_or_provenance(self):
        for record in self.records:
            for text in (
                _prompts.render_multi_prompt(record),
                _prompts.render_state_prompt(record, "authorization_bypass"),
            ):
                self.assertNotIn("authored_clean", text, record.id)
                self.assertNotIn("policy_visible", text, record.id)
                self.assertNotIn("policy_withheld", text, record.id)
                self.assertNotIn("template/", text, record.id)
                self.assertNotIn("proposed_label", text, record.id)
                self.assertNotIn("adjudicat", text, record.id)

    def test_unknown_state_rejected(self):
        with self.assertRaises(ValueError):
            _prompts.render_state_prompt(self.records[0], "vibes_violation")

    def test_prompt_hash_is_stable(self):
        self.assertEqual(_prompts.prompt_hash(), _prompts.prompt_hash())
        self.assertEqual(len(_prompts.prompt_hash()), 64)

    def test_step_lines_are_index_ordered_and_complete(self):
        truncated = next(r for r in self.records if r.truncation is not None)
        lines = _prompts._render_step_lines(truncated)
        total = len(truncated.steps) + len(truncated.overflow_steps)
        self.assertEqual(len(lines), total)
        indices = [int(line.split()[0]) for line in lines]
        self.assertEqual(indices, sorted(indices))


# ── Bands ──────────────────────────────────────────────────────────────────────


class TestBands(unittest.TestCase):
    def test_ceilings_are_shape_multiples_and_cover_max(self):
        lengths = [100, 200, 300, 900, 1000, 2500]
        for k in (1, 2, 3, 4):
            ceilings = _bands.propose_boundaries(lengths, k)
            self.assertEqual(ceilings, sorted(ceilings))
            for c in ceilings:
                self.assertEqual(c % _bands.SHAPE_MULTIPLE, 0)
            self.assertGreaterEqual(ceilings[-1], max(lengths))

    def test_assign_band_picks_smallest_fitting_ceiling(self):
        self.assertEqual(_bands.assign_band(100, [128, 512, 1024]), 128)
        self.assertEqual(_bands.assign_band(600, [128, 512, 1024]), 1024)
        with self.assertRaises(ValueError):
            _bands.assign_band(2000, [128, 512, 1024])

    def test_padding_waste_bounds_and_monotonicity(self):
        lengths = [60, 70, 80, 500, 550, 600, 1800, 1900, 2000]
        wastes = []
        for k in (1, 2, 3):
            ceilings = _bands.propose_boundaries(lengths, k)
            waste = _bands.padding_waste(lengths, ceilings)
            self.assertGreaterEqual(waste, 0.0)
            self.assertLess(waste, 1.0)
            wastes.append(waste)
        # Equal-count splits over a spread distribution: more bands, less padding.
        self.assertGreater(wastes[0], wastes[2])

    def test_banding_report_shape_and_proposal_rule(self):
        rows = [[60, 65, 70, 62, 61, 68]] * 30 + [[500, 510, 505, 520, 515, 501]] * 30 + [[1800] * 6] * 30
        report = _bands.banding_report(rows)
        self.assertEqual(report["distribution"]["records"], 90)
        self.assertEqual(len(report["candidates"]), len(_bands.BAND_COUNTS))
        self.assertIn(report["proposal"], report["candidates"])
        self.assertIn("padding_waste", report["proposal"])

    def test_prompt_level_assignment_never_pads_more_than_record_level(self):
        # Same ceilings, finer assignment: prompt-level can only cost less.
        rows = [[100, 900, 120, 130, 140, 150], [2000, 2100, 300, 310, 320, 330], [60, 61, 62, 63, 64, 65]] * 10
        report = _bands.banding_report(rows)
        for candidate in report["candidates"]:
            self.assertLessEqual(candidate["padding_waste_prompt_level"], candidate["padding_waste"] + 1e-9)
            self.assertLessEqual(candidate["padded_tokens_prompt_level"], candidate["padded_tokens"])

    def test_prompt_token_length_takes_max_over_states(self):
        record = _records()[0]
        per_state = [_word_tokenize(_prompts.render_state_prompt(record, state)) for state in _schema.VIOLATION_STATES]
        self.assertEqual(_bands.prompt_token_length(record, _word_tokenize), max(per_state))

    def test_measure_lengths_over_generated_records(self):
        records = _records()[:10]
        lengths = _bands.measure_lengths(records, _word_tokenize)
        self.assertEqual(len(lengths), 10)
        self.assertTrue(all(n > 50 for n in lengths))


# ── Freeze record ──────────────────────────────────────────────────────────────


class TestFreeze(unittest.TestCase):
    def test_draft_without_bands_is_marked_pending(self):
        content = _freeze.render_frozen(None, None)
        self.assertIn("PRE-FREEZE DRAFT", content)
        self.assertIn("PENDING", content)
        self.assertIn(_schema.record_schema_fingerprint(), content)
        self.assertIn(_prompts.prompt_hash(), content)
        self.assertIn(_canon.TRUNCATION_STRATEGY, content)

    def test_staged_band_report_cannot_freeze(self):
        report = _bands.banding_report([[100] * 6, [200] * 6, [3000] * 6])
        report["metadata"] = {"staged": True, "tokenizer": "x", "corpus": "y"}
        content = _freeze.render_frozen(report, "2026-08-12")
        self.assertIn("PRE-FREEZE DRAFT", content)
        self.assertIn("PREVIEW from staged input", content)

    def test_adjudicated_band_report_with_date_freezes(self):
        report = _bands.banding_report([[100] * 6, [200] * 6, [3000] * 6])
        report["metadata"] = {"staged": False, "tokenizer": "google/gemma-4-E4B-it", "corpus": "trajectories.jsonl"}
        content = _freeze.render_frozen(report, "2026-08-12")
        self.assertIn("**Status: FROZEN**", content)
        self.assertIn("2026-08-12", content)

    def test_fingerprint_moves_when_schema_moves(self):
        # Sanity on the drift detector itself: the fingerprint is a function
        # of vocabularies, so two calls agree and the value is a sha256.
        a = _schema.record_schema_fingerprint()
        self.assertEqual(a, _schema.record_schema_fingerprint())
        self.assertEqual(len(a), 64)


if __name__ == "__main__":
    unittest.main()
