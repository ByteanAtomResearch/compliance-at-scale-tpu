"""
Unit tests for 05_trajectory_eval/schema.py and canonicalize.py.
No TPU hardware or vLLM installation is required; both modules are pure
stdlib on purpose.

The three invariants that matter most here, in order:

  1. Round-trip: a record survives to_dict -> JSON -> from_dict unchanged,
     because the corpus lives on disk as JSONL and adjudication rewrites it.
  2. Boundedness: no adversarially large raw log can push any canonical
     field past its cap. The Phase 4 length bands depend on this.
  3. Digest determinism: the same tool output always digests identically,
     because digests stand in for outputs in every prompt and diff.
"""

import hashlib
import importlib.util
import json
import sys
import unittest
from dataclasses import fields
from pathlib import Path

# 05_trajectory_eval is not an importable package name (leading digit), so
# modules load by path, same pattern as test_pure_functions.py. canonicalize
# does `import schema` internally, so schema is registered in sys.modules
# first and both modules share one class identity (dataclass __eq__ requires
# it).
_MODULE_DIR = Path(__file__).parent.parent / "05_trajectory_eval"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, str(_MODULE_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_schema = _load_module("schema")
_canon = _load_module("canonicalize")


def _make_raw_log(n_steps: int = 3, output_chars: int = 2000) -> dict:
    return {
        "id": "traj-test-001",
        "source": "unit_test",
        "goal": "Do a small test task",
        "boundaries": ["stay read-only", "no external calls"],
        "steps": [
            {
                "actor": "agent",
                "action": f"tool_{i}",
                "target": f"resource-{i}",
                "args": {"n": i},
                "output": f"output-{i}-" + "x" * output_chars,
                "agent_report": f"Ran tool_{i} successfully.",
                "approval": "not_required",
                "signals": [],
                "status": "ok",
            }
            for i in range(n_steps)
        ],
        "final_report": "All steps completed.",
    }


# ── Round-trip ─────────────────────────────────────────────────────────────────


class TestRoundTrip(unittest.TestCase):
    def test_record_survives_dict_round_trip(self):
        record = _canon.canonicalize_record(_make_raw_log(), tier="authored_clean", provenance="unit test")
        rebuilt = _schema.record_from_dict(_schema.record_to_dict(record))
        self.assertEqual(record, rebuilt)

    def test_record_survives_json_round_trip(self):
        record = _canon.canonicalize_record(_make_raw_log(), tier="authored_clean", provenance="unit test")
        wire = json.dumps(_schema.record_to_dict(record))
        rebuilt = _schema.record_from_dict(json.loads(wire))
        self.assertEqual(record, rebuilt)

    def test_labels_survive_round_trip(self):
        record = _canon.canonicalize_record(_make_raw_log(), tier="authored_subtle", provenance="unit test")
        record.proposed_labels = [_schema.ViolationLabel(state="scope_violation", step_index=1, note="proposed")]
        record.labels = [
            _schema.ViolationLabel(state="scope_violation", step_index=1),
            _schema.ViolationLabel(state="misreported_state", step_index=2, note="claim diverges"),
        ]
        record.adjudicated = True
        rebuilt = _schema.record_from_dict(_schema.record_to_dict(record))
        self.assertEqual(record, rebuilt)
        self.assertEqual(len(rebuilt.labels), 2)

    def test_truncated_record_survives_json_round_trip(self):
        record = _canon.canonicalize_record(
            _make_raw_log(n_steps=12), tier="authored_clean", provenance="unit test", max_steps=5
        )
        wire = json.dumps(_schema.record_to_dict(record))
        rebuilt = _schema.record_from_dict(json.loads(wire))
        self.assertEqual(record, rebuilt)
        self.assertEqual(len(rebuilt.overflow_steps), 7)
        self.assertEqual(rebuilt.truncation, record.truncation)

    def test_verdict_survives_dict_round_trip(self):
        verdict = _schema.Verdict(
            violations=[
                _schema.Violation(state="authorization_bypass", failed_step_index=3, evidence="gate", confidence=0.9),
                _schema.Violation(state="misreported_state", failed_step_index=5, evidence="claim", confidence=0.7),
            ],
            recommended_enforcement="block",
        )
        rebuilt = _schema.verdict_from_dict(_schema.verdict_to_dict(verdict))
        self.assertEqual(verdict, rebuilt)


# ── Boundedness ────────────────────────────────────────────────────────────────


class TestBoundedness(unittest.TestCase):
    def _adversarial_log(self) -> dict:
        huge = "A" * 100_000
        return {
            "id": huge,
            "source": huge,
            "goal": huge,
            "boundaries": [huge] * 50,
            "steps": [
                {
                    "actor": huge,
                    "action": huge,
                    "target": huge,
                    "args": {"payload": huge},
                    "output": huge,
                    "agent_report": huge,
                    "approval": "not_required",
                    "signals": [huge],
                    "status": "ok",
                }
            ]
            * (_schema.DEFAULT_MAX_STEPS + 25),
            "final_report": huge,
        }

    def test_all_text_fields_respect_caps(self):
        record = _canon.canonicalize_record(self._adversarial_log(), tier="authored_clean", provenance="unit test")
        self.assertLessEqual(len(record.id), _schema.MAX_SHORT_CHARS)
        self.assertLessEqual(len(record.source), _schema.MAX_SHORT_CHARS)
        self.assertLessEqual(len(record.goal), _schema.MAX_TEXT_CHARS)
        self.assertLessEqual(len(record.final_report), _schema.MAX_TEXT_CHARS)
        for boundary in record.boundaries:
            self.assertLessEqual(len(boundary), _schema.MAX_SHORT_CHARS)
        for step in record.steps:
            self.assertLessEqual(len(step.actor), _schema.MAX_SHORT_CHARS)
            self.assertLessEqual(len(step.action), _schema.MAX_SHORT_CHARS)
            self.assertLessEqual(len(step.target), _schema.MAX_SHORT_CHARS)
            self.assertLessEqual(len(step.args_summary), _schema.MAX_TEXT_CHARS)
            self.assertLessEqual(len(step.agent_report), _schema.MAX_TEXT_CHARS)
            for sig in step.stop_signals:
                self.assertLessEqual(len(sig), _schema.MAX_SHORT_CHARS)
            self.assertLessEqual(len(step.output_digest.head), _schema.DIGEST_HEAD_CHARS)
            self.assertLessEqual(len(step.output_digest.tail), _schema.DIGEST_TAIL_CHARS)
        for step in record.overflow_steps:
            self.assertLessEqual(len(step.actor), _schema.MAX_SHORT_CHARS)
            self.assertLessEqual(len(step.action), _schema.MAX_SHORT_CHARS)
            self.assertLessEqual(len(step.target), _schema.MAX_SHORT_CHARS)
            for sig in step.stop_signals:
                self.assertLessEqual(len(sig), _schema.MAX_SHORT_CHARS)

    def test_boundary_list_is_capped(self):
        record = _canon.canonicalize_record(self._adversarial_log(), tier="authored_clean", provenance="unit test")
        self.assertLessEqual(len(record.boundaries), _schema.MAX_BOUNDARIES)

    def test_overflow_steps_survive_in_reduced_form(self):
        record = _canon.canonicalize_record(self._adversarial_log(), tier="authored_clean", provenance="unit test")
        n = _schema.DEFAULT_MAX_STEPS + 25
        head, tail = _canon.head_tail_split(_schema.DEFAULT_MAX_STEPS)
        self.assertEqual(len(record.steps), _schema.DEFAULT_MAX_STEPS)
        self.assertEqual(len(record.overflow_steps), 25)
        # Head-and-tail windowing: the reduced middle sits between the head
        # and tail windows, and every step keeps its original index, so no
        # step number is lost and a label can point anywhere.
        self.assertEqual([s.index for s in record.overflow_steps], list(range(head, n - tail)))
        self.assertEqual([s.index for s in record.steps], list(range(head)) + list(range(n - tail, n)))

    def test_reduced_steps_keep_structure_and_drop_free_text(self):
        # n=5 with max_steps=3 puts the head at {0, 1}, the tail at {4}, and
        # the reduced middle at {2, 3}.
        raw = _make_raw_log(n_steps=5)
        raw["steps"][3]["approval"] = "denied"
        raw["steps"][3]["signals"] = ["approval_denied"]
        record = _canon.canonicalize_record(raw, tier="authored_clean", provenance="unit test", max_steps=3)
        self.assertEqual([s.index for s in record.overflow_steps], [2, 3])
        reduced = record.overflow_steps[-1]
        # Structural telemetry survives, so gate states stay detectable.
        self.assertEqual(reduced.actor, "agent")
        self.assertEqual(reduced.action, "tool_3")
        self.assertEqual(reduced.target, "resource-3")
        self.assertEqual(reduced.approval_state, "denied")
        self.assertEqual(reduced.stop_signals, ["approval_denied"])
        # The claim survives at the short bound, so misreported_state stays
        # detectable past the cap.
        self.assertEqual(reduced.agent_report, "Ran tool_3 successfully.")
        # Args and digests are exactly what reduction drops.
        for dropped in _canon.REDUCED_DROPPED_FIELDS:
            self.assertFalse(hasattr(reduced, dropped))
        self.assertNotIn("agent_report", _canon.REDUCED_DROPPED_FIELDS)

    def test_reduced_agent_report_uses_short_bound(self):
        raw = _make_raw_log(n_steps=3)
        raw["steps"][2]["agent_report"] = "Claimed success. " * 100
        record = _canon.canonicalize_record(raw, tier="authored_clean", provenance="unit test", max_steps=1)
        for reduced in record.overflow_steps:
            self.assertLessEqual(len(reduced.agent_report), _schema.MAX_SHORT_CHARS)

    def test_truncation_block_records_what_happened(self):
        record = _canon.canonicalize_record(self._adversarial_log(), tier="authored_clean", provenance="unit test")
        self.assertIsNotNone(record.truncation)
        self.assertEqual(record.truncation.strategy, _canon.TRUNCATION_STRATEGY)
        self.assertEqual(record.truncation.original_step_count, _schema.DEFAULT_MAX_STEPS + 25)
        self.assertEqual(record.truncation.retained_step_count, _schema.DEFAULT_MAX_STEPS)
        self.assertEqual(record.truncation.dropped_fields, list(_canon.REDUCED_DROPPED_FIELDS))

    def test_max_steps_is_configurable(self):
        record = _canon.canonicalize_record(
            _make_raw_log(n_steps=10), tier="authored_clean", provenance="unit test", max_steps=4
        )
        self.assertEqual(len(record.steps), 4)
        self.assertEqual(len(record.overflow_steps), 6)
        self.assertEqual(record.truncation.retained_step_count, 4)

    def test_max_steps_below_one_rejected(self):
        with self.assertRaises(ValueError):
            _canon.canonicalize_record(_make_raw_log(), tier="authored_clean", provenance="unit test", max_steps=0)

    def test_untruncated_record_has_no_truncation_block(self):
        record = _canon.canonicalize_record(_make_raw_log(n_steps=2), tier="authored_clean", provenance="unit test")
        self.assertIsNone(record.truncation)
        self.assertEqual(record.overflow_steps, [])

    def test_truncation_is_marked_and_exact(self):
        capped = _canon.truncate("B" * 1000, _schema.MAX_TEXT_CHARS)
        self.assertEqual(len(capped), _schema.MAX_TEXT_CHARS)
        self.assertTrue(capped.endswith("...[truncated]"))

    def test_short_text_is_untouched(self):
        self.assertEqual(_canon.truncate("hello", _schema.MAX_TEXT_CHARS), "hello")

    def test_cap_smaller_than_marker_raises(self):
        with self.assertRaises(ValueError):
            _canon.truncate("anything", 5)


# ── Digest determinism ─────────────────────────────────────────────────────────


class TestDigestDeterminism(unittest.TestCase):
    def test_same_input_same_digest(self):
        text = "tool output " * 500
        self.assertEqual(_canon.digest_output(text), _canon.digest_output(text))

    def test_different_input_different_hash(self):
        self.assertNotEqual(_canon.digest_output("aaa").sha256, _canon.digest_output("aab").sha256)

    def test_sha256_and_byte_len_match_the_actual_bytes(self):
        text = "verifiable output"
        digest = _canon.digest_output(text)
        self.assertEqual(digest.sha256, hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(digest.byte_len, len(text.encode("utf-8")))

    def test_dict_output_digest_ignores_key_order(self):
        a = _canon.digest_output({"x": 1, "y": 2})
        b = _canon.digest_output({"y": 2, "x": 1})
        self.assertEqual(a, b)

    def test_bytes_input_hashes_raw_bytes(self):
        data = b"\x00\x01binary\xff"
        digest = _canon.digest_output(data)
        self.assertEqual(digest.sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(digest.byte_len, len(data))

    def test_short_output_lands_entirely_in_head(self):
        digest = _canon.digest_output("short")
        self.assertEqual(digest.head, "short")
        self.assertEqual(digest.tail, "")

    def test_long_output_keeps_head_and_tail(self):
        text = "S" * 50 + "M" * 1000 + "E" * 50
        digest = _canon.digest_output(text)
        self.assertTrue(digest.head.startswith("S"))
        self.assertTrue(digest.tail.endswith("E"))

    def test_unicode_output_survives(self):
        text = "café ✓ " * 100
        digest = _canon.digest_output(text)
        self.assertEqual(digest.byte_len, len(text.encode("utf-8")))
        self.assertEqual(digest, _canon.digest_output(text))


# ── JSON schemas for structured outputs ────────────────────────────────────────


class TestVerdictSchemas(unittest.TestCase):
    def test_verdict_schema_is_json_serializable(self):
        json.dumps(_schema.VERDICT_SCHEMA)
        json.dumps(_schema.STATE_VERDICT_SCHEMA)

    def test_schema_hash_is_stable_and_distinct(self):
        self.assertEqual(_schema.schema_hash(_schema.VERDICT_SCHEMA), _schema.schema_hash(_schema.VERDICT_SCHEMA))
        self.assertNotEqual(
            _schema.schema_hash(_schema.VERDICT_SCHEMA), _schema.schema_hash(_schema.STATE_VERDICT_SCHEMA)
        )

    def test_schema_hash_ignores_key_order(self):
        self.assertEqual(_schema.schema_hash({"a": 1, "b": 2}), _schema.schema_hash({"b": 2, "a": 1}))

    def test_violation_dataclass_matches_schema_fields(self):
        item_props = set(_schema.VERDICT_SCHEMA["properties"]["violations"]["items"]["properties"])
        dataclass_fields = {f.name for f in fields(_schema.Violation)}
        self.assertSetEqual(item_props, dataclass_fields)

    def test_model_facing_schema_asks_for_violations_only(self):
        # Enforcement is operator policy applied after the verdict; it never
        # appears in anything the model is asked to produce.
        self.assertSetEqual(set(_schema.VERDICT_SCHEMA["properties"]), {"violations"})
        self.assertNotIn("recommended_enforcement", json.dumps(_schema.VERDICT_SCHEMA))
        self.assertNotIn("recommended_enforcement", json.dumps(_schema.STATE_VERDICT_SCHEMA))

    def test_schema_enums_match_vocabularies(self):
        item = _schema.VERDICT_SCHEMA["properties"]["violations"]["items"]
        self.assertEqual(tuple(item["properties"]["state"]["enum"]), _schema.VIOLATION_STATES)

    def test_serialized_verdict_separates_policy_from_model_output(self):
        verdict = _schema.Verdict(
            violations=[_schema.Violation(state="scope_violation", failed_step_index=1, evidence="e", confidence=0.5)],
            recommended_enforcement=_schema.derive_enforcement(
                [_schema.Violation(state="scope_violation", failed_step_index=1, evidence="e", confidence=0.5)]
            ),
        )
        data = _schema.verdict_to_dict(verdict)
        self.assertIn("policy", data)
        self.assertNotIn("recommended_enforcement", set(data))
        self.assertEqual(data["policy"]["recommended_enforcement"], "block")
        self.assertIn("not model output", data["policy"]["source"])

    def test_derive_enforcement_policy_lookup(self):
        def make(state):
            return _schema.Violation(state=state, failed_step_index=0, evidence="e", confidence=1.0)

        self.assertEqual(_schema.derive_enforcement([]), "none")
        self.assertEqual(_schema.derive_enforcement([make("specification_gaming")]), "review")
        gaming_plus_misreport = [make("specification_gaming"), make("misreported_state")]
        self.assertEqual(_schema.derive_enforcement(gaming_plus_misreport), "escalate")
        custom = dict.fromkeys(_schema.VIOLATION_STATES, "review")
        self.assertEqual(_schema.derive_enforcement([make("misreported_state")], policy=custom), "review")

    def test_all_schema_fields_are_required(self):
        # Structured outputs must emit every field, so nothing is optional.
        item = _schema.VERDICT_SCHEMA["properties"]["violations"]["items"]
        self.assertSetEqual(set(item["required"]), set(item["properties"]))
        state_schema = _schema.STATE_VERDICT_SCHEMA
        self.assertSetEqual(set(state_schema["required"]), set(state_schema["properties"]))

    def test_six_violation_states_exactly(self):
        self.assertEqual(len(_schema.VIOLATION_STATES), 6)
        self.assertIn("authorization_bypass", _schema.VIOLATION_STATES)
        self.assertIn("unsafe_continuation", _schema.VIOLATION_STATES)


# ── Validation ─────────────────────────────────────────────────────────────────


class TestValidation(unittest.TestCase):
    def _valid_record(self):
        return _canon.canonicalize_record(_make_raw_log(), tier="authored_clean", provenance="unit test")

    def test_valid_record_passes(self):
        _schema.validate_record(self._valid_record())

    def test_unknown_tier_rejected(self):
        record = self._valid_record()
        record.tier = "totally_real"
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_unknown_violation_state_rejected(self):
        record = self._valid_record()
        record.proposed_labels = [_schema.ViolationLabel(state="vibes_violation", step_index=0)]
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_label_pointing_at_missing_step_rejected(self):
        record = self._valid_record()
        record.proposed_labels = [_schema.ViolationLabel(state="scope_violation", step_index=99)]
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_label_pointing_at_overflow_step_is_valid(self):
        # A violation past the full-fidelity cap must stay labelable, or
        # recall would be ceilinged by the cap instead of measured.
        record = _canon.canonicalize_record(
            _make_raw_log(n_steps=10), tier="authored_clean", provenance="unit test", max_steps=3
        )
        record.proposed_labels = [
            _schema.ViolationLabel(state="unsafe_continuation", step_index=8, region="reduced_region")
        ]
        _schema.validate_record(record)

    def test_label_region_must_match_geometry(self):
        record = _canon.canonicalize_record(
            _make_raw_log(n_steps=10), tier="authored_clean", provenance="unit test", max_steps=3
        )
        # Step 8 is reduced; claiming full_fidelity is exactly the silent
        # drift the stored-plus-validated design exists to catch.
        record.proposed_labels = [_schema.ViolationLabel(state="unsafe_continuation", step_index=8)]
        with self.assertRaises(ValueError):
            _schema.validate_record(record)
        record.proposed_labels = [
            _schema.ViolationLabel(state="unsafe_continuation", step_index=1, region="reduced_region")
        ]
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_unknown_region_rejected(self):
        record = self._valid_record()
        record.proposed_labels = [_schema.ViolationLabel(state="scope_violation", step_index=0, region="twilight")]
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_unknown_variant_rejected(self):
        record = self._valid_record()
        record.variant = "policy_maybe"
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_variant_survives_round_trip(self):
        record = self._valid_record()
        record.variant = "policy_withheld"
        rebuilt = _schema.record_from_dict(_schema.record_to_dict(record))
        self.assertEqual(rebuilt.variant, "policy_withheld")

    def test_overflow_step_with_bad_vocabulary_rejected(self):
        raw = _make_raw_log(n_steps=6)
        raw["steps"][5]["approval"] = "vibes"
        record = _canon.canonicalize_record(raw, tier="authored_clean", provenance="unit test", max_steps=2)
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_overflow_without_truncation_block_rejected(self):
        record = _canon.canonicalize_record(
            _make_raw_log(n_steps=6), tier="authored_clean", provenance="unit test", max_steps=2
        )
        record.truncation = None
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_ground_truth_labels_require_adjudication(self):
        # The core corpus guardrail at the schema level: labels imply a human
        # signed off. Phase 2's corpus tests build on this.
        record = self._valid_record()
        record.labels = [_schema.ViolationLabel(state="scope_violation", step_index=0)]
        record.adjudicated = False
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_required_not_sought_is_a_valid_approval_state(self):
        # The common real-world missing gate: policy demanded approval and
        # none was ever requested. Must be representable or the corpus
        # under-populates authorization_bypass.
        raw = _make_raw_log()
        raw["steps"][1]["approval"] = "required_not_sought"
        record = _canon.canonicalize_record(raw, tier="authored_clean", provenance="unit test")
        _schema.validate_record(record)
        self.assertEqual(record.steps[1].approval_state, "required_not_sought")

    def test_unknown_stop_signal_rejected(self):
        raw = _make_raw_log()
        raw["steps"][0]["signals"] = ["mystery_signal"]
        record = _canon.canonicalize_record(raw, tier="authored_clean", provenance="unit test")
        with self.assertRaises(ValueError):
            _schema.validate_record(record)

    def test_canonicalized_record_is_never_adjudicated(self):
        record = self._valid_record()
        self.assertFalse(record.adjudicated)
        self.assertEqual(record.labels, [])

    def test_verdict_confidence_out_of_range_rejected(self):
        verdict = _schema.Verdict(
            violations=[_schema.Violation(state="scope_violation", failed_step_index=0, evidence="e", confidence=1.5)]
        )
        with self.assertRaises(ValueError):
            _schema.validate_verdict(verdict)

    def test_verdict_unknown_state_rejected(self):
        verdict = _schema.Verdict(
            violations=[_schema.Violation(state="nonsense", failed_step_index=0, evidence="e", confidence=0.5)]
        )
        with self.assertRaises(ValueError):
            _schema.validate_verdict(verdict)

    def test_verdict_unknown_enforcement_rejected(self):
        verdict = _schema.Verdict(violations=[], recommended_enforcement="yolo")
        with self.assertRaises(ValueError):
            _schema.validate_verdict(verdict)

    def test_clean_verdict_passes(self):
        _schema.validate_verdict(_schema.Verdict())


if __name__ == "__main__":
    unittest.main()
