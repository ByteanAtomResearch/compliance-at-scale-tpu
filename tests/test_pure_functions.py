"""
Unit tests for the pure-Python functions in batch_rai_eval.py and
integration_demo.py. No TPU hardware or vLLM installation is required.

These functions -- parse_response, build_prompts, load_records,
aggregate_results, generate_markdown_report, generate_yaml_report -- are
the logic-heavy parts of the tutorial that are straightforward to verify
without any inference infrastructure.

We mock the vllm and jax imports so the tests run in any CI environment,
including the existing ci.yml job that has no vllm-tpu installed.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ── Mock vllm and jax before importing the modules under test ─────────────────
# batch_rai_eval.py imports `from vllm import LLM, SamplingParams` at the top
# level. We substitute MagicMock objects so the import succeeds without those
# packages installed.
for _mod_name in ("vllm", "vllm.sampling_params", "jax"):
    sys.modules.setdefault(_mod_name, MagicMock())


def _load_module(rel_path: str):
    """Load a module by path relative to the project root."""
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(rel_path, str(root / rel_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_batch = _load_module("02_offline_batch/batch_rai_eval.py")
_demo = _load_module("04_integration_demo/integration_demo.py")


# ── parse_response ─────────────────────────────────────────────────────────────


class TestParseResponse(unittest.TestCase):
    def test_valid_json_returned_as_dict(self):
        result = _batch.parse_response('{"detected": true, "types": ["email"], "evidence": "found"}')
        self.assertEqual(result, {"detected": True, "types": ["email"], "evidence": "found"})

    def test_json_in_markdown_fences_with_language_tag(self):
        raw = '```json\n{"detected": false, "types": [], "evidence": "none"}\n```'
        result = _batch.parse_response(raw)
        self.assertNotIn("parse_error", result)
        self.assertFalse(result["detected"])

    def test_json_in_fences_trailing_newline_after_close(self):
        # Regression for M2: closing fence followed by newline was previously
        # left in place, causing json.JSONDecodeError.
        raw = '```json\n{"detected": true}\n```\n'
        result = _batch.parse_response(raw)
        self.assertNotIn("parse_error", result)
        self.assertTrue(result["detected"])

    def test_json_in_plain_fences_no_language_tag(self):
        raw = '```\n{"detected": false}\n```'
        result = _batch.parse_response(raw)
        self.assertNotIn("parse_error", result)

    def test_json_with_preamble_text(self):
        raw = 'Here is the analysis:\n{"detected": true, "types": ["ssn"]}'
        result = _batch.parse_response(raw)
        self.assertNotIn("parse_error", result)
        self.assertTrue(result["detected"])

    def test_invalid_response_returns_parse_error(self):
        result = _batch.parse_response("This is not JSON at all.")
        self.assertTrue(result.get("parse_error"))
        self.assertIn("raw_response", result)

    def test_empty_string_returns_parse_error(self):
        result = _batch.parse_response("")
        self.assertTrue(result.get("parse_error"))


# ── build_prompts ──────────────────────────────────────────────────────────────


class TestBuildPrompts(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "r-01", "text": "Hello world"},
            {"id": "r-02", "text": "Test record"},
        ]

    def test_produces_three_triples_per_record(self):
        triples = _batch.build_prompts(self.records)
        self.assertEqual(len(triples), 6)  # 2 records x 3 heuristics

    def test_triple_contains_record_id_heuristic_and_prompt(self):
        triples = _batch.build_prompts(self.records[:1])
        for record_id, heuristic_name, prompt_text in triples:
            self.assertEqual(record_id, "r-01")
            self.assertIn(heuristic_name, _batch.HEURISTIC_PROMPTS)
            self.assertIn("Hello world", prompt_text)

    def test_all_three_heuristic_names_are_present(self):
        triples = _batch.build_prompts(self.records[:1])
        names = {t[1] for t in triples}
        self.assertSetEqual(names, {"pii_data_leakage", "jailbreak_override", "tone_stereotyping"})

    def test_fifty_records_produce_150_prompts(self):
        records = [{"id": f"r-{i:03d}", "text": f"text {i}"} for i in range(50)]
        self.assertEqual(len(_batch.build_prompts(records)), 150)


# ── load_records ───────────────────────────────────────────────────────────────


class TestLoadRecords(unittest.TestCase):
    def _write_jsonl(self, lines: list[str]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.writelines(lines)
            return f.name

    def test_valid_jsonl_loads_all_records(self):
        path = self._write_jsonl(
            [
                '{"id": "r-01", "text": "hello"}\n',
                '{"id": "r-02", "text": "world"}\n',
            ]
        )
        records = _batch.load_records(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], "r-01")

    def test_malformed_line_is_skipped(self):
        path = self._write_jsonl(
            [
                '{"id": "r-01", "text": "good"}\n',
                "not valid json\n",
                '{"id": "r-02", "text": "also good"}\n',
            ]
        )
        records = _batch.load_records(path)
        self.assertEqual(len(records), 2)
        self.assertEqual({r["id"] for r in records}, {"r-01", "r-02"})

    def test_empty_lines_are_skipped(self):
        path = self._write_jsonl(
            [
                '{"id": "r-01", "text": "hello"}\n',
                "\n",
                '{"id": "r-02", "text": "world"}\n',
            ]
        )
        self.assertEqual(len(_batch.load_records(path)), 2)


# ── aggregate_results ──────────────────────────────────────────────────────────


class TestAggregateResults(unittest.TestCase):
    def _make_records_and_triples(self, texts):
        records = [{"id": f"r-{i:02d}", "text": t, "source": "test"} for i, t in enumerate(texts, 1)]
        triples = _batch.build_prompts(records)
        return records, triples

    def test_all_clean_returns_zero_flagged(self):
        records, triples = self._make_records_and_triples(["hello"])
        raw = [
            '{"detected": false, "types": [], "evidence": "none"}',
            '{"detected": false, "confidence": 0.01, "reasoning": "clean"}',
            '{"detected": false, "categories": [], "severity": "none"}',
        ]
        report = _batch.aggregate_results(records, triples, raw, "test-model")
        for hname in _batch.HEURISTIC_PROMPTS:
            self.assertEqual(report["summary"][hname]["flagged"], 0)
            self.assertEqual(report["summary"][hname]["clean"], 1)
            self.assertEqual(report["summary"][hname]["parse_errors"], 0)

    def test_detected_true_appears_in_flagged_ids(self):
        records, triples = self._make_records_and_triples(["my ssn is 123-45-6789"])
        raw = [
            '{"detected": true, "types": ["ssn"], "evidence": "ssn found"}',
            '{"detected": false, "confidence": 0.01, "reasoning": "clean"}',
            '{"detected": false, "categories": [], "severity": "none"}',
        ]
        report = _batch.aggregate_results(records, triples, raw, "test-model")
        self.assertIn("r-01", report["summary"]["pii_data_leakage"]["flagged_ids"])
        self.assertEqual(report["summary"]["pii_data_leakage"]["flagged"], 1)

    def test_parse_error_counted_not_flagged(self):
        records, triples = self._make_records_and_triples(["hello"])
        raw = [
            "this is not json",  # parse error for pii
            '{"detected": false, "confidence": 0.0, "reasoning": "clean"}',
            '{"detected": false, "categories": [], "severity": "none"}',
        ]
        report = _batch.aggregate_results(records, triples, raw, "test-model")
        pii = report["summary"]["pii_data_leakage"]
        self.assertEqual(pii["parse_errors"], 1)
        self.assertEqual(pii["flagged"], 0)
        self.assertEqual(pii["clean"], 0)  # parse errors don't count as clean

    def test_metadata_and_results_structure(self):
        records, triples = self._make_records_and_triples(["hello", "world"])
        raw = [
            '{"detected": false, "types": [], "evidence": ""}',
            '{"detected": false, "confidence": 0.0, "reasoning": ""}',
            '{"detected": false, "categories": [], "severity": "none"}',
        ] * 2
        report = _batch.aggregate_results(records, triples, raw, "test-model")
        self.assertEqual(report["metadata"]["model"], "test-model")
        self.assertEqual(report["metadata"]["total_records"], 2)
        self.assertEqual(len(report["results"]), 2)
        self.assertIn("text_preview", report["results"][0])


# ── generate_markdown_report ───────────────────────────────────────────────────


class TestGenerateMarkdownReport(unittest.TestCase):
    def _make_report(self, pii_detected=False):
        evaluations = {
            "r-01": {
                "pii_data_leakage": {"detected": pii_detected, "types": [], "evidence": ""},
                "jailbreak_override": {"detected": False, "confidence": 0.0, "reasoning": ""},
                "tone_stereotyping": {"detected": False, "categories": [], "severity": "none"},
            }
        }
        return {"model": "m", "total_records": 1, "elapsed_seconds": 1.0, "evaluations": evaluations}

    def test_clean_heuristic_has_checkmark(self):
        md = _demo.generate_markdown_report(self._make_report(pii_detected=False))
        self.assertIn("[x]", md)
        self.assertIn("All records passed", md)

    def test_flagged_heuristic_has_empty_checkbox(self):
        md = _demo.generate_markdown_report(self._make_report(pii_detected=True))
        self.assertIn("[ ]", md)

    def test_flagged_section_says_issues_found_not_passed(self):
        md = _demo.generate_markdown_report(self._make_report(pii_detected=True))
        self.assertIn("Issues found", md)
        # The flagged section should NOT say "All records passed"
        lines = [line for line in md.splitlines() if "[ ]" in line]
        self.assertTrue(all("All records passed" not in line for line in lines))

    def test_flagged_id_appears_in_output(self):
        md = _demo.generate_markdown_report(self._make_report(pii_detected=True))
        self.assertIn("r-01", md)


# ── generate_yaml_report ───────────────────────────────────────────────────────


class TestGenerateYamlReport(unittest.TestCase):
    def _make_report(self, pii_detected=False):
        evaluations = {
            "r-01": {
                "pii_data_leakage": {"detected": pii_detected},
                "jailbreak_override": {"detected": False},
                "tone_stereotyping": {"detected": False},
            }
        }
        return {"model": "m", "total_records": 1, "elapsed_seconds": 1.0, "evaluations": evaluations}

    def test_pass_status_when_no_flags(self):
        import yaml

        data = yaml.safe_load(_demo.generate_yaml_report(self._make_report(pii_detected=False)))
        for section in data["rai_evaluation"].values():
            self.assertEqual(section["status"], "PASS")

    def test_fail_status_and_count_when_flagged(self):
        import yaml

        data = yaml.safe_load(_demo.generate_yaml_report(self._make_report(pii_detected=True)))
        pii = data["rai_evaluation"]["Pii Data Leakage"]
        self.assertEqual(pii["status"], "FAIL")
        self.assertEqual(pii["flagged_count"], 1)
        self.assertIn("r-01", pii["flagged_records"])

    def test_output_is_valid_yaml(self):
        import yaml

        raw = _demo.generate_yaml_report(self._make_report())
        self.assertIsInstance(yaml.safe_load(raw), dict)


if __name__ == "__main__":
    unittest.main()
