"""
Unit tests for 06_misreported_state_torchtpu/run.py and lab05.py.

No TPU and no vllm. The generate loop is exercised against recording fakes
installed in sys.modules for this test only, and the same fakes are fed to
05's real run_band so the parity assertions compare two live calls, never a
hardcoded copy of 05's values. A test against copied constants would keep
passing after 05 changed.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).parent.parent
_LAB06 = _ROOT / "06_misreported_state_torchtpu"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# lab05 first: run.py does `import lab05` as a sibling import. Then 05
# references through the same loader run.py used, so identities match.
_lab05 = _load("lab05", _LAB06 / "lab05.py")
_run = _load("misreport_run", _LAB06 / "run.py")
_schema = _lab05.load("schema")
_gen = _lab05.load("generate_candidates")
_bands = _lab05.load("bands")
_runner = _lab05.load("batch_trajectory_eval")

_ABSENT = object()
VALID_STATE_RESPONSE = json.dumps({"detected": False, "failed_step_index": -1, "evidence": "", "confidence": 0.0})


# ── Fakes and fixtures ─────────────────────────────────────────────────────────


class _Recorder:
    def __init__(self):
        self.llm_kwargs: list[dict] = []
        self.sampling_kwargs: list[dict] = []
        self.generate_calls: list[tuple[list, object]] = []


def _fake_modules(recorder: _Recorder) -> dict[str, types.ModuleType]:
    """A vllm, vllm.sampling_params and jax that record what they are given.

    Plain module objects, never MagicMock: len(MagicMock()) is 0, which would
    quietly turn the chip count into 0.
    """

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            recorder.sampling_kwargs.append(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            recorder.llm_kwargs.append(kwargs)

        def generate(self, prompts, sampling):
            recorder.generate_calls.append((list(prompts), sampling))
            return [SimpleNamespace(outputs=[SimpleNamespace(text=VALID_STATE_RESPONSE)]) for _ in prompts]

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = FakeStructuredOutputsParams  # type: ignore[attr-defined]
    vllm.sampling_params = sampling_params  # type: ignore[attr-defined]
    jax = types.ModuleType("jax")

    def _no_devices(kind=None):
        raise RuntimeError("no accelerator under test")

    jax.devices = _no_devices  # type: ignore[attr-defined]
    return {"vllm": vllm, "vllm.sampling_params": sampling_params, "jax": jax}


def _install_modules(test: unittest.TestCase, modules: dict) -> None:
    saved = {name: sys.modules.get(name, _ABSENT) for name in modules}

    def restore():
        for name, value in saved.items():
            if value is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    test.addCleanup(restore)
    sys.modules.update(modules)


def _tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


def _adjudicated_corpus(per_state: int = 1, clean: int = 2, seed: int = 11):
    records = _gen.generate_candidates(per_state=per_state, clean=clean, seed=seed)
    for record in records:
        record.labels = list(record.proposed_labels)
        record.adjudicated = True
        record.adjudication_note = "synthetic test corpus 2026-11-08"
        _schema.validate_record(record)
    return records


def _write_corpus(records, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_schema.record_to_dict(record)) + "\n")


def _write_band_report(records, path: Path) -> dict:
    lengths = _bands.measure_state_lengths(records, lambda s: len(s.split()))
    report = _bands.banding_report(lengths)
    report["record_lengths"] = {r.id: max(row) for r, row in zip(records, lengths)}
    report["metadata"] = {"staged": False, "tokenizer": "whitespace-test"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return report


def _args(**over) -> argparse.Namespace:
    base = dict(
        backend="torchtpu",
        dtype="bfloat16",
        model="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
        input="unused.jsonl",
        staged=False,
        bands_report="unused.json",
        band="all",
        limit=None,
        repeat=1,
        compile_only=False,
        eval_set=True,
        single_call=False,
        dry_run=False,
        output=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ── lab05 loader ───────────────────────────────────────────────────────────────


class TestLab05Loader(unittest.TestCase):
    def test_adopts_existing_registration_from_05_file(self):
        self.assertIs(_lab05.load("schema"), sys.modules["schema"])
        self.assertEqual(Path(sys.modules["schema"].__file__).resolve(), _lab05.LAB05_DIR / "schema.py")

    def test_loads_dependencies_before_dependents(self):
        saved = sys.modules.pop("prompts")
        self.addCleanup(lambda: sys.modules.__setitem__("prompts", saved))
        _lab05.load("bands")
        self.assertIn("prompts", sys.modules)
        self.assertEqual(Path(sys.modules["prompts"].__file__).resolve(), _lab05.LAB05_DIR / "prompts.py")

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            _lab05.load("nope")

    def test_failed_exec_leaves_no_registration(self):
        class FailingLoader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise RuntimeError("boom")

        spec = importlib.machinery.ModuleSpec("freeze", FailingLoader(), origin=str(_lab05.path_for("freeze")))
        saved = sys.modules.pop("freeze", _ABSENT)
        self.addCleanup(
            lambda: sys.modules.pop("freeze", None) if saved is _ABSENT else sys.modules.__setitem__("freeze", saved)
        )
        with patch.object(importlib.util, "spec_from_file_location", return_value=spec):
            with self.assertRaises(RuntimeError):
                _lab05.load("freeze")
        self.assertNotIn("freeze", sys.modules)


# ── Generate-loop parity against 05's real run_band ───────────────────────────


class TestGenerateLoopParity(unittest.TestCase):
    def setUp(self):
        self.records = _adjudicated_corpus()
        env = patch.dict(os.environ, {"TPU_CHIPS": "4"})
        env.start()
        self.addCleanup(env.stop)

    def _run_both(self, dtype: str = "bfloat16", **over):
        args = _args(**over)
        rec05 = _Recorder()
        _install_modules(self, _fake_modules(rec05))
        result05 = _runner.run_band(self.records, args, 512, 1024)
        rec06 = _Recorder()
        _install_modules(self, _fake_modules(rec06))
        result06 = _run.run_band(self.records, args, 512, 1024, dtype, 4)
        return rec05, result05, rec06, result06

    def test_sampling_params_identical_field_by_field(self):
        rec05, _, rec06, _ = self._run_both()
        left, right = rec05.sampling_kwargs[0], rec06.sampling_kwargs[0]
        self.assertEqual(set(left), set(right), "SamplingParams keyword sets differ between 05 and 06")
        for field in sorted(set(left) - {"structured_outputs"}):
            self.assertEqual(
                right[field],
                left[field],
                f"SamplingParams.{field}: 06 run.py passed {right[field]!r}, 05 run_band passed {left[field]!r}",
            )

    def test_structured_outputs_schema_identical(self):
        rec05, _, rec06, _ = self._run_both()
        left = rec05.sampling_kwargs[0]["structured_outputs"].kwargs
        right = rec06.sampling_kwargs[0]["structured_outputs"].kwargs
        self.assertEqual(right, left, "StructuredOutputsParams kwargs differ between 05 and 06")
        self.assertEqual(right["json"], _schema.STATE_VERDICT_SCHEMA)

    def test_llm_kwargs_identical_with_default_dtype(self):
        rec05, _, rec06, _ = self._run_both()
        self.assertEqual(rec06.llm_kwargs[0], rec05.llm_kwargs[0])

    def test_llm_kwargs_differ_only_in_dtype_when_requested(self):
        rec05, _, rec06, _ = self._run_both(dtype="auto")
        left, right = rec05.llm_kwargs[0], rec06.llm_kwargs[0]
        differing = {k for k in set(left) | set(right) if left.get(k) != right.get(k)}
        self.assertEqual(differing, {"dtype"})
        self.assertEqual(right["dtype"], "auto")

    def test_same_prompts_reach_generate_in_same_order(self):
        rec05, _, rec06, _ = self._run_both()
        self.assertEqual(rec06.generate_calls[0][0], rec05.generate_calls[0][0])

    def test_repeat_count_honoured_like_05(self):
        rec05, _, rec06, _ = self._run_both(repeat=3)
        self.assertEqual(len(rec05.generate_calls), 3)
        self.assertEqual(len(rec06.generate_calls), 3)

    def test_band_result_shape_matches_05(self):
        _, result05, _, result06 = self._run_both()
        self.assertEqual(set(result06), set(result05))
        self.assertEqual(result06["responses"].keys(), result05["responses"].keys())
        self.assertEqual(result06["prompts"], len(self.records) * 6)

    def test_single_call_mode_uses_verdict_schema_on_both(self):
        rec05, _, rec06, _ = self._run_both(single_call=True)
        self.assertEqual(rec05.sampling_kwargs[0]["structured_outputs"].kwargs["json"], _schema.VERDICT_SCHEMA)
        self.assertEqual(rec06.sampling_kwargs[0]["structured_outputs"].kwargs["json"], _schema.VERDICT_SCHEMA)

    def test_run_band_signature_has_no_backend_parameter(self):
        self.assertNotIn("backend", inspect.signature(_run.run_band).parameters)

    def test_metadata_sampling_params_is_the_object_passed_to_vllm(self):
        rec05, _, rec06, _ = self._run_both()
        captured = {k: v for k, v in rec06.sampling_kwargs[0].items() if k != "structured_outputs"}
        self.assertEqual(captured, _run.SAMPLING)
        self.assertEqual(
            _run.extra_metadata(_args(), 4, {"name": "x", "version": "y"}, "0" * 64)["sampling_params"], _run.SAMPLING
        )


# ── Backend guard ─────────────────────────────────────────────────────────────


class TestBackendGuard(unittest.TestCase):
    def test_torchtpu_distribution_satisfies_torchtpu(self):
        self.assertEqual(
            _run.require_backend("torchtpu", [("vllm-torchtpu", "0.32.0")]),
            {"name": "vllm-torchtpu", "version": "0.32.0"},
        )

    def test_jax_distribution_refuses_torchtpu_naming_both(self):
        with self.assertRaises(SystemExit) as ctx:
            _run.require_backend("torchtpu", [("vllm-tpu", "0.33.0")])
        message = str(ctx.exception)
        self.assertIn("vllm-torchtpu", message)
        self.assertIn("vllm-tpu 0.33.0", message)
        self.assertIn("--backend jax", message)

    def test_neither_installed_refuses(self):
        with self.assertRaises(SystemExit) as ctx:
            _run.require_backend("jax", [])
        self.assertIn("vllm-tpu", str(ctx.exception))
        self.assertIn("vllm-torchtpu", str(ctx.exception))

    def test_cuda_vllm_alone_is_not_a_backend(self):
        with self.assertRaises(SystemExit) as ctx:
            _run.require_backend("torchtpu", [("vllm", "0.11.0")])
        self.assertIn("'vllm' distribution is installed", str(ctx.exception))

    def test_both_installed_refuses_as_ambiguous(self):
        with self.assertRaises(SystemExit) as ctx:
            _run.require_backend("jax", [("vllm-tpu", "1"), ("vllm-torchtpu", "2")])
        self.assertIn("both", str(ctx.exception))

    def test_installed_distributions_reads_importlib_metadata(self):
        def fake_version(name):
            if name == "vllm-torchtpu":
                return "9.9"
            raise importlib.metadata.PackageNotFoundError(name)

        with patch("importlib.metadata.version", side_effect=fake_version):
            self.assertEqual(_run.installed_vllm_distributions(), [("vllm-torchtpu", "9.9")])

    def test_main_refuses_before_reading_input(self):
        def fake_version(name):
            if name == "vllm-tpu":
                return "0.33.0"
            raise importlib.metadata.PackageNotFoundError(name)

        with patch("importlib.metadata.version", side_effect=fake_version):
            with self.assertRaises(SystemExit) as ctx:
                _run.main(_args(backend="torchtpu", input="/nonexistent/never.jsonl", dry_run=False))
        self.assertIn("vllm-tpu 0.33.0", str(ctx.exception))


# ── Dry run ───────────────────────────────────────────────────────────────────


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.records = _adjudicated_corpus(per_state=2, clean=4, seed=9)
        self.corpus_path = _tmp("trajectories.jsonl")
        _write_corpus(self.records, self.corpus_path)
        self.report_path = _tmp("band_report.json")
        _write_band_report(self.records, self.report_path)
        env = patch.dict(os.environ, {"TPU_CHIPS": "4"})
        env.start()
        self.addCleanup(env.stop)

    def _dry_run(self, **over) -> dict:
        out = _tmp("plan06.json")
        _run.main(
            _args(
                input=str(self.corpus_path),
                bands_report=str(self.report_path),
                dry_run=True,
                eval_set=False,
                output=str(out),
                **over,
            )
        )
        return json.loads(out.read_text(encoding="utf-8"))

    def test_dry_run_imports_no_vllm_and_skips_guard(self):
        _install_modules(self, {"vllm": None, "vllm.sampling_params": None, "jax": None})
        with patch("importlib.metadata.version", side_effect=AssertionError("guard must not run under --dry-run")):
            plan = self._dry_run()
        self.assertFalse(plan["executed"])
        self.assertEqual(plan["metadata"]["vllm_distribution"]["name"], "unchecked (dry run)")

    def test_plan_identical_to_05_dry_run(self):
        out05 = _tmp("plan05.json")
        _runner.main(
            argparse.Namespace(
                model="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
                input=str(self.corpus_path),
                staged=False,
                bands_report=str(self.report_path),
                band="all",
                limit=None,
                repeat=1,
                compile_only=False,
                eval_set=False,
                single_call=False,
                dry_run=True,
                output=str(out05),
            )
        )
        plan05 = json.loads(out05.read_text(encoding="utf-8"))
        plan06 = self._dry_run(dtype="auto")
        self.assertEqual(plan06["plan"], plan05["plan"])
        self.assertEqual(plan06["note"], plan05["note"])
        self.assertEqual(sum(e["prompts"] for e in plan06["plan"]), len(self.records) * 6)
        # Every key 05 records is still there, with the same value, except the
        # sampling_params dict 06 now sources from the real call.
        for key, value in plan05["metadata"].items():
            self.assertIn(key, plan06["metadata"])
            if key not in ("sampling_params", "cache"):
                self.assertEqual(plan06["metadata"][key], value, key)
        self.assertEqual(plan06["metadata"]["sampling_params"], plan05["metadata"]["sampling_params"])

    def test_metadata_additions_present(self):
        meta = self._dry_run()["metadata"]
        for key in (
            "backend",
            "dtype",
            "checkpoint",
            "tensor_parallel_size",
            "vllm_distribution",
            "band_report_sha256",
            "git",
            "runner",
            "xla_cache_path_env",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["backend"], "torchtpu")
        self.assertEqual(meta["dtype"], "bfloat16")
        self.assertEqual(meta["checkpoint"], meta["model"])
        self.assertEqual(meta["tensor_parallel_size"], 4)
        self.assertEqual(meta["runner"], "06_misreported_state_torchtpu/run.py")
        self.assertEqual(set(meta["git"]), {"sha", "branch", "dirty"})

    def test_band_report_sha256_matches_file_bytes(self):
        meta = self._dry_run()["metadata"]
        self.assertEqual(meta["band_report_sha256"], hashlib.sha256(self.report_path.read_bytes()).hexdigest())

    def test_dry_run_sampling_params_equal_the_single_source(self):
        self.assertEqual(self._dry_run()["metadata"]["sampling_params"], _run.SAMPLING)

    def test_default_output_is_backend_specific(self):
        self.assertEqual(_run.default_output("jax", True), "results/06_verdicts_jax.json")
        self.assertEqual(_run.default_output("torchtpu", False), "results/06_run_torchtpu.json")


# ── Git provenance ────────────────────────────────────────────────────────────


class TestGitProvenance(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GIT_SHA", None)

    def test_git_missing_falls_back_without_crashing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("git")):
            state = _run.git_provenance(_ROOT)
        self.assertEqual(state, {"sha": _run.GIT_UNAVAILABLE, "branch": _run.GIT_UNAVAILABLE, "dirty": None})

    def test_env_override_wins(self):
        os.environ["GIT_SHA"] = "abc123"
        with patch("subprocess.run", side_effect=FileNotFoundError("git")):
            self.assertEqual(_run.git_provenance(_ROOT)["sha"], "abc123")

    @unittest.skipIf(shutil.which("git") is None, "git not installed")
    def test_not_a_checkout_falls_back(self):
        state = _run.git_provenance(Path(tempfile.mkdtemp()))
        self.assertEqual(state["sha"], _run.GIT_UNAVAILABLE)
        self.assertIsNone(state["dirty"])

    @unittest.skipIf(shutil.which("git") is None, "git not installed")
    def test_real_checkout_reports_sha_branch_and_dirty_flag(self):
        try:
            subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, timeout=10)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            self.skipTest("test tree is not a git checkout")
        state = _run.git_provenance(_ROOT)
        sha = state["sha"].removesuffix("-dirty")
        self.assertEqual(len(sha), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in sha))
        self.assertIsInstance(state["dirty"], bool)
        self.assertEqual(state["sha"].endswith("-dirty"), state["dirty"])


# ── CLI ───────────────────────────────────────────────────────────────────────


class TestCli(unittest.TestCase):
    def test_backend_required_and_restricted(self):
        parser = _run.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dry-run"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--backend", "cuda"])

    def test_dtype_default_and_choices(self):
        parser = _run.build_parser()
        self.assertEqual(parser.parse_args(["--backend", "jax"]).dtype, "bfloat16")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--backend", "jax", "--dtype", "float16"])

    def test_all_05_flags_accepted(self):
        parsed = _run.build_parser().parse_args(
            [
                "--backend",
                "torchtpu",
                "--dtype",
                "auto",
                "--model",
                "m",
                "--input",
                "i.jsonl",
                "--staged",
                "--bands-report",
                "b.json",
                "--band",
                "512",
                "--limit",
                "2",
                "--repeat",
                "3",
                "--compile-only",
                "--eval-set",
                "--single-call",
                "--dry-run",
                "--output",
                "o.json",
            ]
        )
        for flag in (
            "model",
            "input",
            "staged",
            "bands_report",
            "band",
            "limit",
            "repeat",
            "compile_only",
            "eval_set",
            "single_call",
            "dry_run",
            "output",
        ):
            self.assertTrue(hasattr(parsed, flag), flag)
        self.assertEqual(parsed.limit, 2)
        self.assertEqual(parsed.repeat, 3)


if __name__ == "__main__":
    unittest.main()
