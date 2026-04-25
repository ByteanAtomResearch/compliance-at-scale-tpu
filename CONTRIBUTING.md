# Contributing

Thanks for your interest in improving this tutorial. A few quick notes on how to contribute.

## Issues

Open an issue for:

- Bugs in any of the module scripts or the Colab notebook
- Broken links or prose errors in the READMEs
- Outdated API usage (vLLM, `tpu-inference`, or Gemma model names evolve quickly)
- Feature requests for additional heuristics or modules

When filing a bug, include:

1. Which module you ran (`01_setup`, `02_offline_batch`, `03_online_server`, `04_integration_demo`, or the Colab notebook)
2. The TPU type you're on (v5e-4 Docker, bare metal, Colab v2, etc.)
3. The full error output
4. Your `vllm` and `tpu_inference` versions (run `python 01_setup/verify_install.py`)

## Pull requests

- Keep changes focused: one module or one doc fix per PR
- Run `ruff check .` and `ruff format .` before pushing — or install the pre-commit hook so this happens automatically on every commit:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
  After that, `git commit` will run lint and format checks automatically. To run against all files manually: `pre-commit run --all-files`.
- Do not add abstraction layers or wrapper classes to the module scripts. The code is intentionally flat so readers can follow it linearly.
- Do not introduce production features like retry logic, queues, or CLI frameworks. This is a teaching repo.

## Code style

- Type hints on function signatures
- Docstrings that explain the "why," not the "what"
- Rich formatting for user-facing output
- Conversational prose in READMEs. Avoid AI-ish filler phrases.

## Prose conventions

- No em dashes. Use commas, parentheses, or period-breaks instead.
- No negation clauses ("not just X, but Y"). Rewrite positively.
- Prefer short sentences over long ones.

## License

By contributing, you agree your changes will be licensed under Apache 2.0.
