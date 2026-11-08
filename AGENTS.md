# Agent rules for this repository

These rules apply to any coding agent working in this repository. Claude Code
reads this file through `CLAUDE.md`; Codex and Antigravity read it directly.
They are the guardrails Part 3 (module 06) was built under, and they hold for
later work too.

## Standing rules

1. Never modify anything under `05_trajectory_eval/`. Module 06 imports 05 by
   path through `06_misreported_state_torchtpu/lab05.py` and copies none of
   it. `git diff --stat main -- 05_trajectory_eval 04_integration_demo` must
   print nothing at every checkpoint. The one sanctioned change under 05 is
   its own freeze (`freeze.py` regenerating `FROZEN.md`), done on `main` as
   its own commit by the maintainer.
2. Work on the branch `part3-module06`. Never commit to `main` and never push
   to a different branch. Show every diff before committing.
3. Ask before any command that creates, starts or deletes cloud resources
   (VMs, TPUs, buckets). Every VM gets a maximum run duration and is deleted
   at the end of its step.
4. Never run `pip install -e .` in a TorchTPU environment. The lab's
   pyproject pulls in `vllm-tpu`, which would replace TorchTPU's vllm. A
   local test venv installs only `rich` and `pyyaml`.
5. Every step must also work as plain shell commands a reader can run with
   no agent. When you run something, show the exact command.
6. Every run log records the commit SHA. `06_misreported_state_torchtpu/run.py`
   writes `git.sha`, `git.branch` and `git.dirty` into the run metadata, and a
   dirty tree is marked `-dirty`. A number goes into a post only if a results
   file holds it with that metadata.
7. Tags (`v2.0-part2`, `v3.0-part3`) are cut and pushed from the maintainer's
   machine, never from a cloud session. `v2.0-part2` exists before
   `part3-module06` merges to `main`, because `main`'s README tracks `main`
   for the Part 2 badge until that tag exists.

## Files that belong to Part 2's freeze

`results/band_report.json` and `05_trajectory_eval/FROZEN.md` belong to the
Part 2 freeze. Module 06 writes its band reports to other paths and never
touches these two.

## Checks before any commit

```bash
make test                                                    # unittest discover, no TPU, no vllm
ruff check . && ruff format --check .                        # ruff 0.7.0, as CI pins
git diff --stat main -- 05_trajectory_eval 04_integration_demo   # must print nothing
```
