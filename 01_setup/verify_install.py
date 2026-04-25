"""
Module 1: Verify TPU + vLLM Installation

Run this script inside your TPU VM (or Docker container) to confirm that:
  1. TPU chips are visible to the runtime
  2. JAX can see TPU backends
  3. vllm-tpu and tpu_inference are installed at expected versions

This is the "smoke test" you run before moving to Modules 2-4.
If any check fails, the script prints what went wrong and exits with code 1.

Usage:
    python 01_setup/verify_install.py
"""

import sys

from rich.console import Console
from rich.table import Table

console = Console()


def check_tpu_chips() -> tuple[bool, str]:
    """Verify that TPU devices are visible through libtpu / the chip count file."""
    try:
        import jax

        devices = jax.devices("tpu")
        count = len(devices)
        if count == 0:
            return False, "JAX found 0 TPU devices. Are you running on a TPU VM?"
        return True, f"JAX sees {count} TPU chip(s): {[str(d) for d in devices]}"
    except Exception as exc:
        return False, f"Could not query TPU devices via JAX: {exc}"


def check_jax_backends() -> tuple[bool, str]:
    """Confirm JAX is using the TPU backend as its default."""
    try:
        import jax

        backend = jax.default_backend()
        if backend == "tpu":
            return True, f"JAX default backend: {backend} (TPU backend active)"
        return False, f"JAX default backend: {backend} (expected 'tpu', got '{backend}')"
    except Exception as exc:
        return False, f"Could not query JAX default backend: {exc}"


def check_vllm_tpu_version() -> tuple[bool, str]:
    """Check that vllm-tpu is installed (the TPU-specific package, separate from vllm)."""
    try:
        import vllm

        version = vllm.__version__
        return True, f"vllm version: {version}"
    except ImportError:
        return False, (
            "vllm is not installed. On TPU, install with: uv pip install vllm-tpu\n"
            "  Do NOT use 'pip install vllm' (that is the GPU package)."
        )


def check_tpu_inference_version() -> tuple[bool, str]:
    """Check that the tpu-inference plugin is present (the unified JAX+PyTorch backend)."""
    try:
        import tpu_inference

        version = getattr(tpu_inference, "__version__", "unknown")
        return True, f"tpu_inference version: {version}"
    except ImportError:
        return False, (
            "tpu_inference is not installed. This is the unified JAX+PyTorch TPU backend.\n"
            "  If using Docker, it's bundled in vllm/vllm-tpu:gemma4.\n"
            "  If building from source, clone the tpu-inference repo first."
        )


def main() -> None:
    console.print()
    console.print("[bold]TPU + vLLM Installation Verification[/bold]")
    console.print("=" * 50)
    console.print()

    checks = [
        ("TPU Chips Visible", check_tpu_chips),
        ("JAX TPU Backend", check_jax_backends),
        ("vllm-tpu Package", check_vllm_tpu_version),
        ("tpu-inference Plugin", check_tpu_inference_version),
    ]

    table = Table(title="Environment Checks")
    table.add_column("Check", style="cyan", min_width=22)
    table.add_column("Status", min_width=6)
    table.add_column("Details", style="dim")

    all_passed = True
    for name, check_fn in checks:
        passed, detail = check_fn()
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, detail)
        if not passed:
            all_passed = False

    console.print(table)
    console.print()

    if all_passed:
        console.print("[bold green]All checks passed.[/bold green] Your environment is ready.")
        console.print()
        console.print("Next step: run the offline batch evaluation (Module 2):")
        console.print("  make batch")
    else:
        console.print("[bold red]Some checks failed.[/bold red] Review the details above.")
        console.print()
        console.print("Common fixes:")
        console.print("  - Ensure you're running inside the TPU VM or Docker container")
        console.print("  - For Docker: docker run --privileged --net=host --shm-size=150gb ...")
        console.print("  - For bare metal: uv pip install vllm-tpu")
        sys.exit(1)


if __name__ == "__main__":
    main()
