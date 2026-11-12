"""
Unit tests for 06_misreported_state_torchtpu/parity.py.

Fixture verdict files are built in-test with 05's real hashes, so the
metadata gate is exercised against the same field names and values 05's
runner writes. The labelled path uses a synthetic adjudicated corpus from
05's generator, the way tests/test_trajectory_score.py does. No TPU, no vllm.
"""

import argparse
import importlib.util
import json
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


_lab05 = _load("lab05", _LAB06 / "lab05.py")
_parity = _load("misreport_parity", _LAB06 / "parity.py")
_schema = _lab05.load("schema")
_prompts = _lab05.load("prompts")
_gen = _lab05.load("generate_candidates")
_runner = _lab05.load("batch_trajectory_eval")

POLICY_SOURCE = "operator policy lookup (DEFAULT_ENFORCEMENT_POLICY), not model output"


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


def _adjudicated_corpus(per_state: int = 2, clean: int = 4, seed: int = 41):
    records = _gen.generate_candidates(per_state=per_state, clean=clean, seed=seed)
    for record in records:
        record.labels = list(record.proposed_labels)
        record.adjudicated = True
        record.adjudication_note = "synthetic test corpus 2026-11-08"
        _schema.validate_record(record)
    return records


def _write_jsonl(records, path: Path, staged: bool = False) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            doc = _schema.record_to_dict(record)
            if staged:
                doc["labels"], doc["adjudicated"], doc["adjudication_note"] = [], False, ""
            f.write(json.dumps(doc) + "\n")
    return path


def _perfect_judge(records) -> dict:
    verdicts = {}
    for record in records:
        violations = [
            {"state": label.state, "failed_step_index": label.step_index, "evidence": "synthetic", "confidence": 0.9}
            for label in record.labels
        ]
        verdicts[record.id] = {
            "violations": violations,
            "policy": {"recommended_enforcement": "none", "source": POLICY_SOURCE},
        }
    return verdicts


def _meta(**over) -> dict:
    base = {
        "mode": "eval_set",
        "model": "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
        "model_revision": "unpinned (record before publishing)",
        "container_image_digest": "unrecorded (set before publishing)",
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 384},
        "structured_outputs_schema_hash": {
            "six_call": _schema.schema_hash(_schema.STATE_VERDICT_SCHEMA),
            "single_call": _schema.schema_hash(_schema.VERDICT_SCHEMA),
        },
        "record_schema_fingerprint": _schema.record_schema_fingerprint(),
        "prompt_version": _prompts.PROMPT_VERSION,
        "prompt_hash": _prompts.prompt_hash(),
        "call_mode": "six_call",
        "band_ceilings_selected": [512],
        "max_model_len": 1024,
        "band_report_sha256": "f" * 64,
        "backend": "jax",
        "dtype": "bfloat16",
        "parse_errors": 0,
        "evaluator": "gemma_judge",
        "git": {"sha": "abc", "branch": "part3-module06", "dirty": False},
    }
    base.update(over)
    return base


def _write_run(name: str, verdicts: dict, meta: dict, wall_times=(1.0,)) -> Path:
    path = _tmp(name)
    doc = {
        "metadata": meta,
        "bands": [
            {
                "ceiling": 512,
                "records": len(verdicts),
                "prompts": len(verdicts) * 6,
                "wall_times_seconds": list(wall_times),
            }
        ],
        "verdicts": verdicts,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _args(**over) -> argparse.Namespace:
    base = dict(left=None, right=None, left_name=None, right_name=None, labels=None, allow_staged=False, output=None)
    base.update(over)
    return argparse.Namespace(**base)


def _compare(left: Path, right: Path, **over) -> dict:
    out = _tmp("parity.json")
    _parity.main(_args(left=str(left), right=str(right), output=str(out), **over))
    return json.loads(out.read_text(encoding="utf-8"))


# ── Kappa ──────────────────────────────────────────────────────────────────────


class TestKappa(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertEqual(_parity.cohens_kappa(5, 0, 0, 5), 1.0)

    def test_chance_level_is_zero(self):
        self.assertEqual(_parity.cohens_kappa(3, 3, 3, 3), 0.0)

    def test_undefined_when_both_unanimous(self):
        self.assertIsNone(_parity.cohens_kappa(0, 0, 0, 9))
        self.assertIsNone(_parity.cohens_kappa(9, 0, 0, 0))
        self.assertIsNone(_parity.cohens_kappa(0, 0, 0, 0))

    def test_one_side_unanimous_is_defined_zero(self):
        self.assertEqual(_parity.cohens_kappa(0, 0, 3, 7), 0.0)

    def test_matches_textbook_formula(self):
        a, b, c, d = 20, 5, 10, 15
        n = a + b + c + d
        p_o = (a + d) / n
        p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
        self.assertAlmostEqual(_parity.cohens_kappa(a, b, c, d), (p_o - p_e) / (1 - p_e))


# ── Metadata gate ─────────────────────────────────────────────────────────────


class TestMetadataGate(unittest.TestCase):
    def setUp(self):
        self.records = _adjudicated_corpus()
        self.verdicts = _perfect_judge(self.records)

    def _pair(self, left_over: dict | None = None, right_over: dict | None = None):
        left = _write_run("left.json", self.verdicts, _meta(**(left_over or {})))
        right = _write_run("right.json", self.verdicts, _meta(**(right_over or {})))
        return left, right

    def test_identical_metadata_passes(self):
        self.assertEqual(_parity.metadata_mismatches(_meta(), dict(_meta())), [])

    def test_missing_verdicts_key_exits_1_not_2(self):
        plan = _tmp("plan.json")
        plan.write_text(json.dumps({"metadata": _meta(), "plan": []}), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            _parity.main(_args(left=str(plan), right=str(plan), output=str(_tmp("out.json"))))
        self.assertNotEqual(ctx.exception.code, 2)
        self.assertIn("verdicts", str(ctx.exception))

    def test_prompt_hash_mismatch_exits_2_and_writes_nothing(self):
        left, right = self._pair(right_over={"prompt_hash": "tampered"})
        out = _tmp("should_not_exist.json")
        with self.assertRaises(SystemExit) as ctx:
            _parity.main(_args(left=str(left), right=str(right), output=str(out)))
        self.assertEqual(ctx.exception.code, _parity.EXIT_REFUSED)
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("prompt_hash", ctx.exception.message)
        self.assertFalse(out.exists())

    def test_every_mismatch_is_named(self):
        right = _meta(
            model="other",
            call_mode="single_call",
            sampling_params={"temperature": 0.7, "top_p": 1.0, "max_tokens": 384},
        )
        message = "\n".join(_parity.metadata_mismatches(_meta(), right))
        for field in ("model", "call_mode", "sampling_params"):
            self.assertIn(field, message)
        self.assertNotIn("prompt_hash", message)

    def test_missing_required_field_is_a_mismatch(self):
        right = _meta()
        del right["record_schema_fingerprint"]
        message = "\n".join(_parity.metadata_mismatches(_meta(), right))
        self.assertIn("record_schema_fingerprint", message)
        self.assertIn("<missing>", message)

    def test_band_report_sha_and_dtype_compared_only_when_both_present(self):
        left = _meta()
        right_without = _meta()
        del right_without["band_report_sha256"]
        del right_without["dtype"]
        self.assertEqual(_parity.metadata_mismatches(left, right_without), [])
        right_different = _meta(band_report_sha256="0" * 64, dtype="auto")
        message = "\n".join(_parity.metadata_mismatches(left, right_different))
        self.assertIn("band_report_sha256", message)
        self.assertIn("dtype", message)
        compared = _parity.compared_metadata(left, right_without)
        self.assertEqual(compared["band_report_sha256"], _parity.NOT_COMPARED_ONE_SIDE)

    def test_max_model_len_is_guarded(self):
        message = "\n".join(_parity.metadata_mismatches(_meta(), _meta(max_model_len=2048)))
        self.assertIn("max_model_len", message)

    def test_backend_and_git_differences_are_informational(self):
        left, right = self._pair(
            right_over={"backend": "torchtpu", "git": {"sha": "def", "branch": "x", "dirty": True}}
        )
        result = _compare(left, right)
        self.assertEqual(result["per_side"]["left"]["informational"]["backend"], "jax")
        self.assertEqual(result["per_side"]["right"]["informational"]["backend"], "torchtpu")
        self.assertEqual(result["per_side"]["right"]["informational"]["git"]["sha"], "def")
        self.assertEqual(result["left"]["name"], "jax")
        self.assertEqual(result["right"]["name"], "torchtpu")

    def test_guard_fields_exist_in_05_metadata(self):
        args = argparse.Namespace(model="m", single_call=False)
        emitted = _runner.build_run_metadata(args, [512], 1024, "eval_set")
        for field in _parity.REQUIRED_MATCH:
            self.assertIn(field, emitted, f"guarded field {field} is not something 05 records")


class TestIdGate(unittest.TestCase):
    def test_mismatched_id_sets_refuse_naming_ids(self):
        records = _adjudicated_corpus()
        verdicts = _perfect_judge(records)
        dropped = records[0].id
        smaller = {rid: v for rid, v in verdicts.items() if rid != dropped}
        left = _write_run("left.json", verdicts, _meta())
        right = _write_run("right.json", smaller, _meta())
        out = _tmp("should_not_exist.json")
        with self.assertRaises(SystemExit) as ctx:
            _parity.main(_args(left=str(left), right=str(right), output=str(out)))
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn(dropped, ctx.exception.message)
        self.assertFalse(out.exists())


# ── Agreement ─────────────────────────────────────────────────────────────────


class TestAgreement(unittest.TestCase):
    def setUp(self):
        self.records = _adjudicated_corpus()
        self.verdicts = _perfect_judge(self.records)

    def test_identical_pair_agrees_100_percent(self):
        left = _write_run("left.json", self.verdicts, _meta())
        right = _write_run("right.json", self.verdicts, _meta(backend="torchtpu"))
        result = _compare(left, right)
        for state, entry in result["agreement"].items():
            self.assertEqual(entry["n"], len(self.records), state)
            self.assertEqual(entry["agree"], entry["n"], state)
            self.assertEqual(entry["agreement_rate"], 1.0, state)
            self.assertEqual(entry["disagree_ids"], [], state)
            if entry["kappa"] is None:
                self.assertEqual(entry["kappa_note"], _parity.KAPPA_UNDEFINED_NOTE)
            else:
                self.assertEqual(entry["kappa"], 1.0, state)
        self.assertEqual(result["overall"]["decisions"], len(self.records) * 6)
        self.assertEqual(result["overall"]["agreement_rate"], 1.0)
        self.assertTrue(result["records"]["ids_identical"])

    def test_single_disagreement_lands_on_that_state_only(self):
        clean_id = next(r.id for r in self.records if not r.labels)
        right_verdicts = json.loads(json.dumps(self.verdicts))
        right_verdicts[clean_id]["violations"].append(
            {"state": "scope_violation", "failed_step_index": 0, "evidence": "phantom", "confidence": 0.5}
        )
        left = _write_run("left.json", self.verdicts, _meta())
        right = _write_run("right.json", right_verdicts, _meta())
        result = _compare(left, right)
        n = len(self.records)
        scope = result["agreement"]["scope_violation"]
        self.assertEqual(scope["agree"], n - 1)
        self.assertEqual(scope["right_only"], 1)
        self.assertEqual(scope["left_only"], 0)
        self.assertEqual(scope["disagree_ids"], [clean_id])
        for state, entry in result["agreement"].items():
            if state != "scope_violation":
                self.assertEqual(entry["agree"], n, state)
        self.assertEqual(result["overall"]["agree"], n * 6 - 1)

    def test_small_sample_and_sparse_positive_flags(self):
        small = _adjudicated_corpus(per_state=1, clean=1, seed=3)  # 7 records
        entry = _parity.state_agreement(
            _perfect_judge(small), _perfect_judge(small), [r.id for r in small], "scope_violation"
        )
        self.assertTrue(entry["small_sample"])
        self.assertIn(_parity.SMALL_SAMPLE_NOTE, entry["notes"])
        # 16 records, 2 positives per state: not small, but positives are sparse.
        entry = _parity.state_agreement(self.verdicts, self.verdicts, [r.id for r in self.records], "scope_violation")
        self.assertFalse(entry["small_sample"])
        self.assertTrue(entry["sparse_positives"])
        self.assertIn(_parity.SPARSE_NOTE, entry["notes"])
        # Balanced 2x2 at n=20 trips neither flag.
        balanced = _parity.cohens_kappa(5, 5, 5, 5)
        self.assertEqual(balanced, 0.0)

    def test_wilson_interval_rides_with_every_rate(self):
        left = _write_run("left.json", self.verdicts, _meta())
        right = _write_run("right.json", self.verdicts, _meta())
        result = _compare(left, right)
        for entry in result["agreement"].values():
            lo, hi = entry["agreement_wilson"]
            self.assertTrue(0.0 <= lo <= hi <= 1.0)
        lo, hi = result["overall"]["agreement_wilson"]
        self.assertTrue(0.0 <= lo <= hi <= 1.0)

    def test_side_summary_carries_parse_errors_sentinel_malformation_provenance(self):
        verdicts = json.loads(json.dumps(self.verdicts))
        some_id = self.records[0].id
        verdicts[some_id]["violations"].append(
            {"state": "misreported_state", "failed_step_index": -1, "evidence": "unlocalized", "confidence": 0.4}
        )
        doc = {"metadata": _meta(parse_errors=3), "bands": [], "verdicts": verdicts}
        with_records = _parity.side_summary(doc, self.records)
        self.assertEqual(with_records["parse_errors"], 3)
        self.assertEqual(with_records["sentinel_unlocalized"], 1)
        self.assertEqual(set(with_records["malformation"]), {"total_violations", "malformed_violations", "rate"})
        self.assertFalse(with_records["provenance_recorded"])
        without_records = _parity.side_summary(doc, None)
        self.assertIsNone(without_records["malformation"])
        pinned = _parity.side_summary(
            {"metadata": _meta(model_revision="sha1", container_image_digest="sha256:x"), "verdicts": {}}, None
        )
        self.assertTrue(pinned["provenance_recorded"])

    def test_band_wall_times_reported_not_compared(self):
        left = _write_run("left.json", self.verdicts, _meta(), wall_times=(1.0,))
        right = _write_run("right.json", self.verdicts, _meta(backend="torchtpu"), wall_times=(2.5, 2.4))
        result = _compare(left, right)
        self.assertEqual(result["per_side"]["left"]["bands"][0]["wall_times_seconds"], [1.0])
        self.assertEqual(result["per_side"]["right"]["bands"][0]["wall_times_seconds"], [2.5, 2.4])
        self.assertEqual(result["per_side"]["right"]["bands"][0]["ceiling"], 512)


# ── Labels ────────────────────────────────────────────────────────────────────


class TestLabels(unittest.TestCase):
    def setUp(self):
        # Eight per state leaves at least five full-fidelity positives in every state, enough to carry a rate.
        self.records = _adjudicated_corpus(per_state=8, clean=6, seed=13)
        self.verdicts = _perfect_judge(self.records)
        self.corpus_path = _write_jsonl(self.records, _tmp("trajectories.jsonl"))

    def test_labels_reports_per_tier_precision_recall_and_delta(self):
        # Drop one full-fidelity detection on the right so recall falls on exactly that state.
        target = next((r, lb) for r in self.records for lb in r.labels if lb.region == "full_fidelity")
        record, label = target
        right_verdicts = json.loads(json.dumps(self.verdicts))
        right_verdicts[record.id]["violations"] = [
            v for v in right_verdicts[record.id]["violations"] if v["state"] != label.state
        ]
        left = _write_run("left.json", self.verdicts, _meta())
        right = _write_run("right.json", right_verdicts, _meta(backend="torchtpu"))
        result = _compare(left, right, labels=str(self.corpus_path))
        tiers = result["labels"]["tiers"]
        self.assertEqual(list(tiers), ["authored_clean"])
        per_state = tiers["authored_clean"]
        self.assertLess(per_state[label.state]["delta"]["recall"], 0.0)
        for state, entry in per_state.items():
            self.assertEqual(set(entry), {"left", "right", "delta", "counts_only", "malformed_pairs", "reduced_region"})
            if state != label.state:
                self.assertEqual(entry["delta"]["recall"], 0.0 if entry["left"]["recall"] is not None else None, state)
        self.assertEqual(result["labels"]["source"], "adjudicated")
        self.assertEqual(result["labels"]["excluded_unverdicted_ids"], [])
        self.assertIn("policy_withheld", result["labels"]["variant_recall"]["left"])
        self.assertEqual(result["labels"]["tier_notes"]["authored_clean"]["counts_only"], False)
        self.assertEqual(result["labels"]["excluded_no_label_ids"], [])

    def test_a_state_with_under_five_positives_is_counts_only(self):
        records = _adjudicated_corpus(per_state=3, clean=6, seed=13)
        verdicts = _perfect_judge(records)
        corpus_path = _write_jsonl(records, _tmp("small.jsonl"))
        left = _write_run("left.json", verdicts, _meta())
        right = _write_run("right.json", verdicts, _meta(backend="torchtpu"))
        labels = _compare(left, right, labels=str(corpus_path))["labels"]
        note = labels["tier_notes"]["authored_clean"]
        self.assertTrue(note["counts_only"])
        self.assertEqual(note["note"], _parity.COUNTS_ONLY_NOTE)
        sparse = [state for state, entry in labels["tiers"]["authored_clean"].items() if entry["counts_only"]]
        self.assertTrue(sparse)
        for state in sparse:
            entry = labels["tiers"]["authored_clean"][state]
            self.assertLess(entry["left"]["positives"], _parity.SPARSE_POSITIVES)
            self.assertIsNone(entry["left"]["recall"])
            self.assertIsNone(entry["right"]["precision_wilson"])
            self.assertEqual(entry["delta"], {"recall": None, "precision": None})
            self.assertEqual(entry["left"]["tp"], entry["left"]["positives"])

    def test_allow_staged_uses_proposed_labels(self):
        staged_path = _write_jsonl(self.records, _tmp("staged.jsonl"), staged=True)
        left = _write_run("left.json", self.verdicts, _meta())
        right = _write_run("right.json", self.verdicts, _meta())
        with self.assertRaises(SystemExit) as ctx:
            _compare(left, right, labels=str(staged_path))
        self.assertNotEqual(ctx.exception.code, 2)  # a labels error, not a parity refusal
        self.assertIn("--allow-staged", str(ctx.exception))
        result = _compare(left, right, labels=str(staged_path), allow_staged=True)
        # 05's generator built these records from templates and nobody reviewed them.
        self.assertEqual(
            result["labels"]["source"],
            f"tier authored_clean, {len(self.records)} records: "
            "template-construction labels, not adjudicated (--allow-staged)",
        )
        self.assertEqual(
            result["labels"]["sources"]["by_tier"]["authored_clean"]["sources"],
            {"template-construction labels, not adjudicated": len(self.records)},
        )
        self.assertIn("authored_clean", result["labels"]["tiers"])
        # The proposals were scored: a perfect judge on proposed labels has full recall.
        recalls = {entry["left"]["recall"] for entry in result["labels"]["tiers"]["authored_clean"].values()}
        self.assertEqual(recalls, {1.0})

    def test_verdict_id_without_label_record_refuses(self):
        verdicts = dict(self.verdicts)
        verdicts["ghost"] = {"violations": [], "policy": {"recommended_enforcement": "none", "source": POLICY_SOURCE}}
        left = _write_run("left.json", verdicts, _meta())
        right = _write_run("right.json", verdicts, _meta())
        with self.assertRaises(SystemExit) as ctx:
            _compare(left, right, labels=str(self.corpus_path))
        self.assertNotEqual(ctx.exception.code, 2)  # a labels error, not a parity refusal
        self.assertIn("ghost", str(ctx.exception))

    def test_labelled_records_without_verdicts_are_excluded_and_listed(self):
        dropped = sorted(r.id for r in self.records[:2])
        smaller = {rid: v for rid, v in self.verdicts.items() if rid not in dropped}
        left = _write_run("left.json", smaller, _meta())
        right = _write_run("right.json", smaller, _meta())
        result = _compare(left, right, labels=str(self.corpus_path))
        self.assertEqual(result["labels"]["excluded_unverdicted_ids"], dropped)
        self.assertEqual(result["records"]["n"], len(self.records) - 2)


# ── Output ────────────────────────────────────────────────────────────────────


class TestOutput(unittest.TestCase):
    def test_output_json_has_expected_top_level_keys(self):
        records = _adjudicated_corpus()
        verdicts = _perfect_judge(records)
        left = _write_run("left.json", verdicts, _meta())
        right = _write_run("right.json", verdicts, _meta())
        result = _compare(left, right, left_name="a", right_name="b")
        expected = {
            "left",
            "right",
            "compared_metadata",
            "records",
            "per_side",
            "agreement",
            "overall",
            "labels",
            "note",
        }
        self.assertTrue(expected <= set(result))
        self.assertEqual(result["left"]["name"], "a")
        self.assertIsNone(result["labels"])
        self.assertEqual(list(result["agreement"]), list(_schema.VIOLATION_STATES))
        self.assertEqual(
            set(result["compared_metadata"]) - set(_parity.REQUIRED_MATCH), set(_parity.MATCH_WHEN_BOTH_PRESENT)
        )


if __name__ == "__main__":
    unittest.main()
