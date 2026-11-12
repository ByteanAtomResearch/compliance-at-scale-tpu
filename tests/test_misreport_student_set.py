"""
Unit tests for module 06's student eval set and its label-source wording.

Three things are pinned here. The 14-record sample is balanced, committed
and rebuildable like the review files. student_set.py folds a reviewer's
decisions over the eight tamper records and refuses anything it cannot
stand behind. And every table and metadata block that names labels uses
labels.py's exact words: "student-adjudicated" for records a person
promoted, "template-construction labels, not adjudicated" for the sample.

The adjudication is simulated through 05's own promote_record, so the files
have the shape adjudicate.py writes. No TPU, no vllm.
"""

import argparse
import datetime
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
_STAGING = _LAB06 / "data" / "staging"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# lab05 first: the 06 scripts import it as a sibling.
_lab05 = _load("lab05", _LAB06 / "lab05.py")
_labels = _lab05.sibling("labels")
_review = _load("misreport_build_review_files", _LAB06 / "build_review_files.py")
_student = _load("misreport_student_set", _LAB06 / "student_set.py")
_run = _load("misreport_run", _LAB06 / "run.py")
_parity = _load("misreport_parity", _LAB06 / "parity.py")
_bridge = _load("misreport_bridge", _LAB06 / "bridge.py")
_readers = _load("misreport_compare_readers", _LAB06 / "compare_readers.py")
_schema = _lab05.load("schema")
_corpus = _lab05.load("corpus")
_bands = _lab05.load("bands")
_rules = _lab05.load("rules_baseline")
_adjudicate = _lab05.load("adjudicate")

STUDENT = "student-adjudicated"
TEMPLATE = "template-construction labels, not adjudicated"
PROPOSED = "proposed labels, not adjudicated"
REJECTED = "rejected at adjudication, no label"
TAMPER_TIER, SAMPLE_TIER = "authored_subtle", "authored_clean"
POLICY_SOURCE = "operator policy lookup (DEFAULT_ENFORCEMENT_POLICY), not model output"
RATE_WORDS = ("rate", "recall", "precision", "kappa", "wilson", "specificity")


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp())


def _stamp(name: str) -> str:
    return f"{name} {datetime.date.today().isoformat()}"


def _tamper() -> list:
    return _corpus.load_staged(_STAGING / _student.TAMPER_FILENAME)


def _review_the_eight(
    work: Path, clean: tuple[str, ...] = (), reject: tuple[str, ...] = (), skip: tuple[str, ...] = (), name: str = "ada"
) -> tuple[Path, Path]:
    """Decide the eight the way adjudicate.py would: accept or clean, with overrides by id suffix."""
    decisions, rejections = work / "tamper_labels.jsonl", work / "tamper_rejections.jsonl"
    for record in _tamper():
        suffix = record.id[-3:]
        if suffix in skip:
            continue
        if suffix in reject:
            record.adjudication_note = f"{_stamp(name)} rejected: not convinced"
            with open(rejections, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(_schema.record_to_dict(record), separators=(",", ":")) + "\n")
            continue
        accept = bool(record.proposed_labels) and suffix not in clean
        record.labels = list(record.proposed_labels) if accept else []
        record.adjudicated = True
        record.adjudication_note = f"{_stamp(name)} {'accepted proposal' if accept else 'adjudicated clean'}"
        _adjudicate.promote_record(record, decisions)
    return decisions, rejections


def _build(work: Path, allow_undecided: bool = False, **review) -> tuple[Path, dict]:
    decisions, rejections = _review_the_eight(work, **review)
    out = work / "eval_set.jsonl"
    result = _student.build(
        _student.DEFAULT_TAMPER, _student.DEFAULT_SAMPLE, decisions, rejections, out, allow_undecided=allow_undecided
    )
    return out, result


def _band_report(records: list, path: Path) -> Path:
    lengths = _bands.measure_state_lengths(records, lambda text: len(text.split()))
    report = _bands.banding_report(lengths)
    report["record_lengths"] = {record.id: max(row) for record, row in zip(records, lengths)}
    report["metadata"] = {"staged": True, "tokenizer": "whitespace-test"}
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _dry_run(eval_set: Path, work: Path) -> dict:
    """A real 06 dry run over the eval set, for the metadata a TPU run would record."""
    bands = _band_report(_corpus.load_staged(eval_set), work / "bands.json")
    out = work / "plan.json"
    argv = ["--backend", "torchtpu", "--staged", "--input", str(eval_set), "--bands-report", str(bands)]
    _run.main(_run.build_parser().parse_args([*argv, "--dry-run", "--output", str(out)]))
    return json.loads(out.read_text(encoding="utf-8"))


def _verdicts_from_labels(records: list) -> dict:
    """A judge that says exactly what each record's scoring labels say."""
    verdicts = {}
    for record in records:
        labels = _labels.scoring_labels(record) or []
        verdicts[record.id] = {
            "violations": [
                {"state": lb.state, "failed_step_index": lb.step_index, "evidence": "fixture", "confidence": 0.9}
                for lb in labels
            ],
            "policy": {"recommended_enforcement": "none", "source": POLICY_SOURCE},
        }
    return verdicts


def _write_run(path: Path, plan: dict, verdicts: dict, backend: str) -> Path:
    metadata = {**plan["metadata"], "mode": "eval_set", "parse_errors": 0, "evaluator": "test_fixture"}
    metadata["backend"] = backend
    path.write_text(json.dumps({"metadata": metadata, "bands": [], "verdicts": verdicts}), encoding="utf-8")
    return path


def _keys(node) -> set[str]:
    """Every dict key anywhere in a JSON document."""
    if isinstance(node, dict):
        return set(node) | {key for value in node.values() for key in _keys(value)}
    if isinstance(node, list):
        return {key for value in node for key in _keys(value)}
    return set()


# ── The 14-record sample ──────────────────────────────────────────────────────


class TestStudentSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = _corpus.load_staged(_STAGING / _review.STUDENT14_FILENAME)

    def test_id_list_is_14_distinct_ids_outside_the_review_files(self):
        self.assertEqual(len(_review.STUDENT14_IDS), 14)
        self.assertEqual(len(set(_review.STUDENT14_IDS)), 14)
        self.assertFalse(set(_review.STUDENT14_IDS) & (set(_review.CUT60_IDS) | set(_review.TOPUP14_IDS)))

    def test_committed_file_matches_a_rebuild(self):
        self.assertIn(_review.STUDENT14_FILENAME, _review.REVIEW_FILES)
        self.assertEqual(_review.check(_review.DEFAULT_SOURCE, _STAGING), [])
        self.assertEqual({record.id for record in self.sample}, set(_review.STUDENT14_IDS))

    def test_two_proposed_positives_per_state_and_two_clean(self):
        per_state = {state: 0 for state in _schema.VIOLATION_STATES}
        clean = 0
        for record in self.sample:
            states = {label.state for label in record.proposed_labels}
            self.assertLessEqual(len(states), 1, record.id)
            if not states:
                clean += 1
            for state in states:
                per_state[state] += 1
        self.assertEqual(per_state, {state: 2 for state in _schema.VIOLATION_STATES})
        self.assertEqual(clean, 2)

    def test_every_record_is_an_unreviewed_full_fidelity_template_record(self):
        for record in self.sample:
            self.assertEqual(record.source, _labels.TEMPLATE_SOURCE, record.id)
            self.assertEqual(record.tier, SAMPLE_TIER, record.id)
            self.assertFalse(record.adjudicated, record.id)
            self.assertIsNone(record.truncation, record.id)
            self.assertLessEqual(len(record.steps), 9, record.id)

    def test_the_bypass_pair_is_one_trajectory_in_both_policy_forms(self):
        pair = [r for r in self.sample if any(lb.state == "authorization_bypass" for lb in r.proposed_labels)]
        self.assertEqual(sorted(r.variant for r in pair), ["policy_visible", "policy_withheld"])
        self.assertEqual(len({r.id.rsplit("-", 1)[0] for r in pair}), 1)


# ── The words ─────────────────────────────────────────────────────────────────


class TestLabelWording(unittest.TestCase):
    def test_the_two_phrases_the_student_track_uses(self):
        self.assertEqual(_labels.LABELS_STUDENT, "student-adjudicated")
        self.assertEqual(_labels.LABELS_TEMPLATE, "template-construction labels, not adjudicated")

    def test_no_phrase_claims_more_than_it_knows(self):
        self.assertEqual(_labels.LABELS_PROPOSED, "proposed labels, not adjudicated")
        self.assertEqual(_labels.LABELS_REJECTED, "rejected at adjudication, no label")
        self.assertEqual(_labels.LABELS_CORPUS, "adjudicated")
        for phrase in (_labels.LABELS_TEMPLATE, _labels.LABELS_PROPOSED):
            self.assertIn("not adjudicated", phrase)

    def test_each_kind_of_record_gets_its_phrase(self):
        out, _ = _build(_tmp_dir(), allow_undecided=True, reject=("005",), skip=("008",))
        by_id = {record.id: record for record in _corpus.load_staged(out)}
        self.assertEqual(_labels.label_source(by_id["traj-t3-tmp-001"], staged=True), STUDENT)
        self.assertEqual(_labels.label_source(by_id["traj-t3-tmp-005"], staged=True), REJECTED)
        self.assertEqual(_labels.label_source(by_id["traj-t3-tmp-008"], staged=True), PROPOSED)
        self.assertEqual(_labels.label_source(by_id["traj-t2-sv-009"], staged=True), TEMPLATE)
        self.assertEqual(_labels.label_source(by_id["traj-t3-tmp-001"], staged=False), "adjudicated")

    def test_an_adjudicated_clean_record_is_scored_as_clean_not_as_its_proposal(self):
        out, _ = _build(_tmp_dir(), clean=("003",))
        record = next(r for r in _corpus.load_staged(out) if r.id == "traj-t3-tmp-003")
        self.assertTrue(record.proposed_labels)
        self.assertEqual(_labels.scoring_labels(record), [])
        # 05's --allow-staged rule would fall back to the proposal here.
        self.assertEqual(record.labels or record.proposed_labels, record.proposed_labels)

    def test_rejected_has_no_label_and_undecided_has_its_proposal(self):
        out, _ = _build(_tmp_dir(), allow_undecided=True, reject=("005",), skip=("004",))
        by_id = {record.id: record for record in _corpus.load_staged(out)}
        self.assertIsNone(_labels.scoring_labels(by_id["traj-t3-tmp-005"]))
        self.assertEqual(_labels.scoring_labels(by_id["traj-t3-tmp-004"]), by_id["traj-t3-tmp-004"].proposed_labels)

    def test_statement_for_a_fully_reviewed_student_set(self):
        _, result = _build(_tmp_dir())
        summary = result["label_sources"]
        self.assertEqual(
            summary["statement"],
            f"tier {SAMPLE_TIER}, 14 records: {TEMPLATE}; tier {TAMPER_TIER}, 8 records: {STUDENT} (by ada)",
        )
        self.assertEqual(summary["by_tier"][TAMPER_TIER]["sources"], {STUDENT: 8})
        self.assertEqual(summary["by_tier"][SAMPLE_TIER]["sources"], {TEMPLATE: 14})
        self.assertEqual(summary["by_tier"][TAMPER_TIER]["adjudicators"], ["ada"])
        self.assertEqual(summary["mode"], "staged")
        self.assertFalse(summary["read_by_judge"])

    def test_statement_never_says_student_adjudicated_for_a_record_nobody_decided(self):
        _, result = _build(_tmp_dir(), allow_undecided=True, skip=tuple(f"00{n}" for n in range(1, 9)))
        tier = result["label_sources"]["by_tier"][TAMPER_TIER]
        self.assertEqual(tier["sources"], {PROPOSED: 8})
        self.assertNotIn(STUDENT, result["label_sources"]["statement"])

    def test_mixed_tier_lists_every_source_with_its_count(self):
        _, result = _build(_tmp_dir(), allow_undecided=True, reject=("005",), skip=("008",))
        text = result["label_sources"]["by_tier"][TAMPER_TIER]["labels"]
        self.assertEqual(text, f"6 {STUDENT} (by ada) + 1 {PROPOSED} + 1 {REJECTED}")


# ── The builder ───────────────────────────────────────────────────────────────


class TestBuilder(unittest.TestCase):
    def test_eight_reviewed_then_the_sample_in_order(self):
        out, result = _build(_tmp_dir())
        records = _corpus.load_staged(out)
        self.assertEqual(len(records), 22)
        self.assertEqual([r.id for r in records[:8]], [f"traj-t3-tmp-00{n}" for n in range(1, 9)])
        self.assertEqual({r.id for r in records[8:]}, set(_review.STUDENT14_IDS))
        self.assertTrue(all(r.adjudicated for r in records[:8]))
        self.assertFalse(any(r.adjudicated for r in records[8:]))
        self.assertEqual(result["status"]["undecided"], [])
        self.assertEqual((result["tamper"], result["sample"], result["records"]), (8, 14, 22))

    def test_decisions_change_review_fields_only(self):
        out, _ = _build(_tmp_dir(), clean=("002",))
        built = {r.id: r for r in _corpus.load_staged(out)}
        staged = _tamper() + _corpus.load_staged(_student.DEFAULT_SAMPLE)
        for record in staged:
            self.assertEqual(_student._trajectory(built[record.id]), _student._trajectory(record), record.id)

    def test_staged_inputs_are_never_written(self):
        paths = (_student.DEFAULT_TAMPER, _student.DEFAULT_SAMPLE)
        before = [path.read_bytes() for path in paths]
        _build(_tmp_dir())
        self.assertEqual([path.read_bytes() for path in paths], before)

    def test_undecided_records_refuse_with_exit_2_and_write_nothing(self):
        work = _tmp_dir()
        with self.assertRaises(SystemExit) as ctx:
            _build(work, skip=("004", "005"))
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("traj-t3-tmp-004", ctx.exception.message)
        self.assertIn("2 of 8", ctx.exception.message)
        self.assertFalse((work / "eval_set.jsonl").exists())

    def test_no_review_at_all_refuses(self):
        work = _tmp_dir()
        with self.assertRaises(SystemExit) as ctx:
            _student.build(
                _student.DEFAULT_TAMPER, _student.DEFAULT_SAMPLE, work / "none.jsonl", work / "none_r.jsonl", work / "o"
            )
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("8 of 8", ctx.exception.message)

    def test_allow_undecided_builds_and_marks_them(self):
        out, result = _build(_tmp_dir(), allow_undecided=True, skip=("004",))
        self.assertEqual(result["status"]["undecided"], ["traj-t3-tmp-004"])
        record = next(r for r in _corpus.load_staged(out) if r.id == "traj-t3-tmp-004")
        self.assertFalse(record.adjudicated)
        self.assertEqual(_labels.label_source(record, staged=True), PROPOSED)

    def test_rejected_record_stays_in_the_set_with_its_reason(self):
        out, result = _build(_tmp_dir(), reject=("005",))
        self.assertEqual(result["status"]["rejected"], ["traj-t3-tmp-005"])
        record = next(r for r in _corpus.load_staged(out) if r.id == "traj-t3-tmp-005")
        self.assertFalse(record.adjudicated)
        self.assertIn("rejected: not convinced", record.adjudication_note)

    def test_decision_on_a_different_draft_is_refused(self):
        work = _tmp_dir()
        decisions, rejections = _review_the_eight(work)
        lines = decisions.read_text(encoding="utf-8").splitlines()
        doc = json.loads(lines[0])
        doc["final_report"] += " Edited after review."
        decisions.write_text("\n".join([json.dumps(doc), *lines[1:]]) + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            _student.build(_student.DEFAULT_TAMPER, _student.DEFAULT_SAMPLE, decisions, rejections, work / "out.jsonl")
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("different draft", ctx.exception.message)
        self.assertFalse((work / "out.jsonl").exists())

    def test_decision_on_a_stranger_record_is_refused(self):
        work = _tmp_dir()
        decisions, rejections = _review_the_eight(work)
        stranger = _corpus.load_staged(_student.DEFAULT_SAMPLE)[0]
        stranger.labels, stranger.adjudicated = list(stranger.proposed_labels), True
        stranger.adjudication_note = f"{_stamp('ada')} accepted proposal"
        _adjudicate.promote_record(stranger, decisions)
        with self.assertRaises(SystemExit) as ctx:
            _student.build(_student.DEFAULT_TAMPER, _student.DEFAULT_SAMPLE, decisions, rejections, work / "out.jsonl")
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn(stranger.id, ctx.exception.message)

    def test_a_staged_file_is_not_a_decisions_file(self):
        work = _tmp_dir()
        with self.assertRaises(SystemExit) as ctx:
            _student.build(
                _student.DEFAULT_TAMPER, _student.DEFAULT_SAMPLE, _student.DEFAULT_TAMPER, work / "r", work / "out"
            )
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("not an adjudicate.py corpus file", ctx.exception.message)

    def test_make_target_runs_the_builder_on_the_student_paths(self):
        makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
        recipe = makefile.split("\nmisreport-student-set:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("python $(MISREPORT_DIR)/student_set.py", recipe)
        for flag in ("--decisions", "--rejections", "--output"):
            self.assertIn(flag, recipe)
        self.assertIn("misreport-student-set", makefile.split(".PHONY:", 1)[1].split("\n\n", 1)[0])
        self.assertEqual(_student.DEFAULT_OUTPUT, "results/student/eval_set.jsonl")


# ── run.py, parity.py and bridge.py on the set ────────────────────────────────


class TestToolsOnTheStudentSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = _tmp_dir()
        # 003 is adjudicated clean against its proposal, 005 is rejected: the two cases 05's rule gets wrong.
        cls.eval_set, cls.built = _build(cls.work, clean=("003",), reject=("005",))
        cls.records = _corpus.load_staged(cls.eval_set)
        cls.plan = _dry_run(cls.eval_set, cls.work)
        left = _verdicts_from_labels(cls.records)
        right = json.loads(json.dumps(left))
        # The right side flags 003 the way its overruled proposal would.
        right["traj-t3-tmp-003"]["violations"] = [
            {"state": "specification_gaming", "failed_step_index": 3, "evidence": "fixture", "confidence": 0.9}
        ]
        cls.left_path = _write_run(cls.work / "jax.json", cls.plan, left, "jax")
        cls.right_path = _write_run(cls.work / "torchtpu.json", cls.plan, right, "torchtpu")

    def _parity(self, **over) -> dict:
        out = _tmp_dir() / "parity.json"
        args = dict(left=str(self.left_path), right=str(self.right_path), left_name="jax", right_name="torchtpu")
        args.update(labels=str(self.eval_set), allow_staged=True, output=str(out))
        args.update(over)
        _parity.main(argparse.Namespace(**args))
        return json.loads(out.read_text(encoding="utf-8"))

    def test_run_staged_records_label_sources_in_its_metadata(self):
        sources = self.plan["metadata"]["label_sources"]
        self.assertEqual(sources["by_tier"][TAMPER_TIER]["sources"], {STUDENT: 7, REJECTED: 1})
        self.assertEqual(sources["by_tier"][SAMPLE_TIER]["sources"], {TEMPLATE: 14})
        self.assertFalse(sources["read_by_judge"])
        self.assertEqual(sum(entry["prompts"] for entry in self.plan["plan"]), 22 * 6)

    def test_run_without_staged_refuses_the_mixed_file(self):
        argv = ["--backend", "torchtpu", "--input", str(self.eval_set), "--bands-report", str(self.work / "bands.json")]
        with self.assertRaises(_corpus.CorpusIntegrityError):
            _run.main(_run.build_parser().parse_args([*argv, "--dry-run", "--output", str(_tmp_dir() / "p.json")]))

    def test_label_sources_are_informational_in_parity_never_a_guard(self):
        self.assertIn("label_sources", _parity.INFORMATIONAL)
        self.assertNotIn("label_sources", _parity.REQUIRED_MATCH + _parity.MATCH_WHEN_BOTH_PRESENT)

    def test_parity_allow_staged_names_whose_labels_each_tier_used(self):
        labels = self._parity()["labels"]
        self.assertEqual(labels["sources"]["by_tier"][TAMPER_TIER]["sources"], {STUDENT: 7, REJECTED: 1})
        self.assertEqual(labels["sources"]["by_tier"][SAMPLE_TIER]["labels"], TEMPLATE)
        self.assertTrue(labels["source"].endswith("(--allow-staged)"))
        self.assertIn(f"tier {SAMPLE_TIER}, 14 records: {TEMPLATE}", labels["source"])
        self.assertIn(f"7 {STUDENT} (by ada) + 1 {REJECTED}", labels["source"])

    def test_parity_without_allow_staged_points_at_the_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            self._parity(allow_staged=False)
        self.assertNotEqual(ctx.exception.code, 2)
        self.assertIn("--allow-staged", str(ctx.exception))

    def test_parity_reports_the_eight_as_counts_with_no_rate_anywhere(self):
        labels = self._parity()["labels"]
        note = labels["tier_notes"][TAMPER_TIER]
        self.assertTrue(note["counts_only"])
        self.assertEqual(note["note"], _parity.COUNTS_ONLY_NOTE)
        for state, entry in labels["tiers"][TAMPER_TIER].items():
            for side in ("left", "right"):
                for field in ("recall", "recall_wilson", "precision", "precision_wilson"):
                    self.assertIsNone(entry[side][field], f"{state} {side} {field}")
                self.assertIsInstance(entry[side]["tp"], int)
                self.assertIsNone(entry["reduced_region"][side]["recall"])
            self.assertEqual(entry["delta"], {"recall": None, "precision": None})
            self.assertTrue(entry["counts_only"])

    def test_parity_gives_no_rate_on_two_positives_either(self):
        labels = self._parity()["labels"]
        self.assertTrue(labels["tier_notes"][SAMPLE_TIER]["counts_only"])
        self.assertEqual(labels["tier_notes"][SAMPLE_TIER]["records"], 14)
        scope = labels["tiers"][SAMPLE_TIER]["scope_violation"]
        self.assertTrue(scope["counts_only"])
        self.assertEqual((scope["left"]["positives"], scope["left"]["tp"], scope["left"]["recall"]), (2, 2, None))

    def test_parity_scores_the_reviewers_clean_decision_not_the_proposal(self):
        gaming = self._parity()["labels"]["tiers"][TAMPER_TIER]["specification_gaming"]
        self.assertEqual(gaming["left"]["positives"], 0)
        self.assertEqual((gaming["left"]["fp"], gaming["right"]["fp"]), (0, 1))

    def test_parity_leaves_the_rejected_record_out_of_the_labelled_tables(self):
        labels = self._parity()["labels"]
        self.assertEqual(labels["excluded_no_label_ids"], ["traj-t3-tmp-005"])
        self.assertEqual(labels["tier_notes"][TAMPER_TIER]["records"], 7)
        # It is still one of the records the two backends are compared on.
        self.assertEqual(self._parity()["records"]["n"], 22)

    def test_parity_blanks_variant_recall_on_one_positive(self):
        for side in ("left", "right"):
            entry = self._parity()["labels"]["variant_recall"][side]["policy_withheld"]
            self.assertEqual((entry["positives"], entry["recall"], entry["wilson"]), (1, None, None))
            self.assertTrue(entry["counts_only"])

    def _bridge(self) -> tuple[str, dict]:
        md_path, yaml_path = _tmp_dir() / "report.md", _tmp_dir() / "report.yaml"
        _bridge.main(argparse.Namespace(verdicts=str(self.right_path), markdown=str(md_path), yaml=str(yaml_path)))
        return md_path.read_text(encoding="utf-8"), yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    def test_bridge_carries_the_run_logs_label_statement(self):
        md, report = self._bridge()
        statement = self.plan["metadata"]["label_sources"]["statement"]
        self.assertEqual(report["provenance"]["label_sources"], statement)
        self.assertIn(f"- label_sources: {statement}", md)
        self.assertIn(_bridge.LABELS_SENTENCE, md)
        self.assertEqual(list(report), ["rai_evaluation", "policy", "provenance"])

    def test_bridge_says_unrecorded_for_a_run_log_without_the_block(self):
        self.assertEqual(_bridge.label_statement({"model": "x"}), _bridge.UNRECORDED)
        self.assertEqual(_bridge.provenance_block({})["label_sources"], _bridge.UNRECORDED)


# ── Three readers ─────────────────────────────────────────────────────────────


class TestCompareReaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = _tmp_dir()
        cls.eval_set, _ = _build(cls.work, clean=("003",), reject=("005",))
        cls.records = _corpus.load_staged(cls.eval_set)
        cls.plan = _dry_run(cls.eval_set, cls.work)
        cls.rules_path = cls.work / "rules.json"
        cls.rules_path.write_text(
            json.dumps({"metadata": {"evaluator": "rules_baseline"}, "verdicts": _rules.evaluate_records(cls.records)}),
            encoding="utf-8",
        )
        judge = _verdicts_from_labels(cls.records)
        judge["traj-t3-tmp-004"]["violations"][0]["failed_step_index"] = 3  # right state, another step
        judge["traj-t3-tmp-002"]["violations"] = []  # a miss
        cls.judge_path = _write_run(cls.work / "judge.json", cls.plan, judge, "torchtpu")

    def _compare(self, **over) -> dict:
        out = _tmp_dir() / "readers.json"
        args = dict(records=str(self.eval_set), allow_staged=True, rules=str(self.rules_path))
        args.update(judge=str(self.judge_path), output=str(out))
        args.update(over)
        _readers.main(argparse.Namespace(**args))
        return json.loads(out.read_text(encoding="utf-8"))

    def test_one_row_per_record_in_two_tiers(self):
        result = self._compare()
        self.assertEqual(len(result["cases"]), 22)
        self.assertEqual(list(result["counts"]), [SAMPLE_TIER, TAMPER_TIER])
        self.assertEqual(result["counts"][TAMPER_TIER]["records"], 8)
        self.assertEqual(result["counts"][SAMPLE_TIER]["records"], 14)

    def test_every_case_and_the_metadata_say_whose_labels(self):
        result = self._compare()
        by_id = {row["id"]: row for row in result["cases"]}
        self.assertEqual(by_id["traj-t3-tmp-001"]["label_source"], STUDENT)
        self.assertEqual(by_id["traj-t3-tmp-005"]["label_source"], REJECTED)
        self.assertEqual(by_id["traj-t2-mr-153"]["label_source"], TEMPLATE)
        sources = result["metadata"]["label_sources"]["by_tier"]
        self.assertEqual(sources[SAMPLE_TIER]["labels"], TEMPLATE)
        self.assertEqual(sources[TAMPER_TIER]["labels"], f"7 {STUDENT} (by ada) + 1 {REJECTED}")

    def test_no_rate_anywhere_in_the_output(self):
        keys = _keys(self._compare())
        offenders = sorted(key for key in keys if any(word in key.lower() for word in RATE_WORDS))
        self.assertEqual(offenders, [])
        self.assertIn("never a rate", self._compare()["metadata"]["note"])

    def test_the_rules_flag_one_of_the_tamper_records(self):
        tamper = [row for row in self._compare()["cases"] if row["tier"] == TAMPER_TIER]
        flagged = {row["id"]: row["rules"] for row in tamper if row["rules"]}
        self.assertEqual(flagged, {"traj-t3-tmp-001": [{"state": "authorization_bypass", "step": 3}]})
        self.assertEqual(self._compare()["counts"][TAMPER_TIER]["rules_flagged_any"], 1)

    def test_miss_extra_and_step_difference_are_told_apart(self):
        by_id = {row["id"]: row for row in self._compare()["cases"]}
        self.assertEqual(by_id["traj-t3-tmp-002"]["judge_vs_labels"]["missed"], ["scope_violation"])
        self.assertFalse(by_id["traj-t3-tmp-002"]["judge_vs_labels"]["same_states"])
        differs = by_id["traj-t3-tmp-004"]["judge_vs_labels"]
        self.assertTrue(differs["same_states"])
        self.assertEqual(
            differs["step_differs"], [{"state": "misreported_state", "label_steps": [2], "reader_steps": [3]}]
        )
        # The rules flag one of 001's two labelled states.
        self.assertEqual(by_id["traj-t3-tmp-001"]["rules_vs_labels"]["missed"], ["scope_violation"])

    def test_a_clean_decision_is_shown_as_clean_and_a_rejection_as_no_label(self):
        by_id = {row["id"]: row for row in self._compare()["cases"]}
        self.assertEqual(by_id["traj-t3-tmp-003"]["labels"], [])
        self.assertIsNone(by_id["traj-t3-tmp-005"]["labels"])
        self.assertIsNone(by_id["traj-t3-tmp-005"]["judge_vs_labels"])
        counts = self._compare()["counts"][TAMPER_TIER]
        self.assertEqual((counts["records"], counts["with_label"]), (8, 7))
        self.assertEqual(counts["judge_same_states_as_labels"], 6)

    def test_a_missing_verdict_refuses_with_exit_2_and_writes_nothing(self):
        doc = json.loads(self.judge_path.read_text(encoding="utf-8"))
        del doc["verdicts"]["traj-t3-tmp-007"]
        short = self.work / "judge_short.json"
        short.write_text(json.dumps(doc), encoding="utf-8")
        out = _tmp_dir() / "readers.json"
        with self.assertRaises(SystemExit) as ctx:
            self._compare(judge=str(short), output=str(out))
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("traj-t3-tmp-007", ctx.exception.message)
        self.assertFalse(out.exists())

    def test_without_allow_staged_points_at_the_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            self._compare(allow_staged=False)
        self.assertNotEqual(ctx.exception.code, 2)
        self.assertIn("--allow-staged", str(ctx.exception))


# ── The student track as written ──────────────────────────────────────────────


class TestStudentTrackDoc(unittest.TestCase):
    """The README is the published lab. These pin its shape, and that its commands name things that exist."""

    STEPS = (
        "setup and laptop rehearsal",
        "meet the eight records and the rules baseline",
        "adjudicate the eight yourself",
        "create the VM, install, smoke check",
        "run the judge on TorchTPU",
        "compare three readers of the same records",
        "the same eval set on the JAX backend, then parity",
        "bridge to the rai-checklist-cli report",
        "tear down the VM and confirm nothing is running",
    )

    @classmethod
    def setUpClass(cls):
        cls.readme = (_LAB06 / "README.md").read_text(encoding="utf-8")
        cls.sections = {heading.split("\n", 1)[0]: heading for heading in cls.readme.split("\n## ")[1:]}
        track = cls.sections["Student track"]
        cls.steps = track.split("\n### ")[1:]

    def test_student_track_is_the_first_path_after_the_front_matter(self):
        order = list(self.sections)
        self.assertEqual(order[:4], ["What you will learn", "Before you start", "Time and cost", "Student track"])
        self.assertLess(order.index("Student track"), order.index("Module map"))
        self.assertEqual(order[-1], "Appendix: author track (optional)")

    def test_nine_steps_in_the_stated_order(self):
        titles = [step.split("\n", 1)[0] for step in self.steps]
        self.assertEqual(titles, [f"Step {n}: {title}" for n, title in enumerate(self.STEPS, start=1)])

    def test_every_step_has_predict_command_output_and_one_checkpoint(self):
        for step in self.steps:
            title = step.split("\n", 1)[0]
            self.assertEqual(step.count("**Predict:**"), 1, title)
            self.assertEqual(step.count("**Checkpoint:**"), 1, title)
            self.assertIn("```bash", step, title)
            self.assertTrue("**Observed" in step or "**Expect" in step, title)
            self.assertLess(step.index("**Predict:**"), step.index("```bash"), title)
            self.assertGreater(step.index("**Checkpoint:**"), step.rindex("```"), title)
            self.assertTrue("`laptop`" in step or "`VM`" in step, title)

    def test_five_questions_and_three_extensions(self):
        explain = self.sections["What you should be able to explain"]
        self.assertEqual(
            [line[:2] for line in explain.splitlines() if line[:2] in {f"{n}." for n in range(1, 10)}],
            ["1.", "2.", "3.", "4.", "5."],
        )
        extensions = self.sections["Extension exercises"]
        self.assertEqual(sum(1 for line in extensions.splitlines() if line[:3] in ("1. ", "2. ", "3. ", "4. ")), 3)

    def test_cost_is_a_formula_with_the_machine_type_and_no_quoted_price(self):
        cost = self.sections["Time and cost"]
        self.assertIn("ct6e-standard-4t", cost)
        self.assertIn("hours", cost)
        self.assertIn("about 90 minutes", cost)
        self.assertNotIn("$", cost)

    def test_label_phrases_and_no_rates_on_eight_are_stated(self):
        track = self.sections["Student track"]
        self.assertIn(_labels.LABELS_STUDENT, track)
        self.assertIn(_labels.LABELS_TEMPLATE, track)
        self.assertIn("No rates on eight records.", track)

    def test_every_script_and_staged_file_the_track_names_exists(self):
        track = self.sections["Student track"]
        paths = set(re.findall(r"(0[56]_[a-z_]+/[A-Za-z0-9_/.]+\.(?:py|jsonl))", track))
        self.assertGreater(len(paths), 8)
        for path in sorted(paths):
            self.assertTrue((_ROOT / path).exists(), path)

    def test_every_make_target_the_track_names_exists(self):
        makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
        targets = set(re.findall(r"^make ([a-z-]+)", self.sections["Student track"], flags=re.MULTILINE))
        self.assertEqual(targets, {"test", "misreport-rehearsal", "misreport-student-set"})
        for target in targets:
            self.assertIn(f"\n{target}:", makefile)

    def test_every_flag_the_track_passes_is_one_the_tool_takes(self):
        track = self.sections["Student track"]
        parsers = {
            "run.py": _run.build_parser(),
            "parity.py": _parity.build_parser(),
            "bridge.py": _bridge.build_parser(),
            "compare_readers.py": _readers.build_parser(),
            "student_set.py": _student.build_parser(),
        }
        for script, parser in parsers.items():
            known = {flag for action in parser._actions for flag in action.option_strings}
            for block in re.findall(r"```bash\n(.*?)```", track, flags=re.DOTALL):
                for command in re.split(
                    r"\n(?=python |sudo |make |gcloud |export |curl |uv |git |mkdir |ls |head |grep |sed |exit)", block
                ):
                    if f"06_misreported_state_torchtpu/{script}" not in command:
                        continue
                    after = command.split(f"06_misreported_state_torchtpu/{script}", 1)[1].split(" | ")[0]
                    after = after.split("2>&1")[0].split(";")[0]
                    for flag in re.findall(r"(--[a-z][a-z-]*)", after):
                        self.assertIn(flag, known, f"{script}: {flag}")

    def test_runlog_template_has_one_block_per_step_with_the_five_fields(self):
        runlog = (_LAB06 / "RUNLOG.md").read_text(encoding="utf-8")
        blocks = [block for block in runlog.split("\n## ")[1:] if block.startswith("Step ")]
        titles = [block.split("\n", 1)[0] for block in blocks]
        self.assertEqual(titles, [f"Step {n}: {title}" for n, title in enumerate(self.STEPS, start=1)])
        for block in blocks:
            title = block.split("\n", 1)[0]
            for field in ("**Command run:**", "**Pass / fail:**", "**Output:**", "**Time taken:**", "**Notes:**"):
                self.assertEqual(block.count(field), 1, f"{title}: {field}")
            self.assertIn("```text", block, title)
        self.assertIn("[RUNLOG.md](RUNLOG.md)", self.sections["Student track"])

    def test_runlog_template_is_empty_and_holds_no_number(self):
        runlog = (_LAB06 / "RUNLOG.md").read_text(encoding="utf-8")
        self.assertIn("Never paste a Hugging Face token", runlog)
        for block in runlog.split("\n## ")[1:]:
            if block.startswith("Step "):
                self.assertRegex(block, r"\*\*Pass / fail:\*\*\n")
                self.assertIn("```text\n\n```", block)

    def test_student_outputs_never_point_at_published_or_the_part_2_corpus(self):
        track = self.sections["Student track"]
        self.assertNotIn("published/", track)
        self.assertNotIn("sample_data/trajectories.jsonl", track)
        self.assertNotIn("freeze.py", track)


if __name__ == "__main__":
    unittest.main()
