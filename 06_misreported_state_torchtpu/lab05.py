"""
Module 6: Load module 05 by path, once, under its own names.

Why this file exists. 05_trajectory_eval/ starts with a digit, so it cannot
be imported as a package, and its modules import each other by bare name
(`from schema import ...`, `from corpus import ...`). Loading one of them
with importlib therefore needs its dependencies registered in sys.modules
under those bare names first, in dependency order. Module 06 copies no 05
code; everything it needs from 05 comes through load().

Why identity matters. 05's dataclasses compare by class identity. If
schema.py ran twice, a TrajectoryRecord from one copy would never equal one
from the other, and validate_record would reject records it should accept.
load() therefore adopts an existing sys.modules entry when it already points
at 05's file (which is what the test suite's own loaders leave behind) and
executes the file from disk only otherwise. Nothing here touches sys.path:
putting 05 on the path would shadow any installed package called `schema`
and would leak into every other module in the process.

Test contract. The 06 scripts do `import lab05` as a sibling import, which
resolves when they run as scripts. A test that loads a 06 script by path must
first load this file by path and register it as sys.modules["lab05"].

06's own shared code. sibling() loads another 06 script by path the same
way, once per process, so run.py, parity.py and bridge.py can share one
module (labels.py) without a second sibling import that every test would
have to register by hand.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB05_DIR = REPO_ROOT / "05_trajectory_eval"
LAB06_DIR = Path(__file__).resolve().parent
SIBLING_PREFIX = "lab06_"

# Each module's bare sibling imports, verified against the import lines on
# main at ced0c80. A name missing here is a KeyError on purpose.
_DEPENDS: dict[str, tuple[str, ...]] = {
    "schema": (),
    "canonicalize": ("schema",),
    "corpus": ("schema",),
    "prompts": ("schema",),
    "generate_candidates": ("schema", "canonicalize"),
    "bands": ("schema", "corpus", "prompts"),
    "rules_baseline": ("schema", "corpus"),
    "batch_trajectory_eval": ("schema", "corpus", "prompts"),
    "score": ("schema", "corpus"),
    "freeze": ("schema", "canonicalize", "prompts"),
    "adjudicate": ("schema", "corpus"),
}


def path_for(name: str) -> Path:
    """The 05 file behind a module name. Unknown names raise KeyError."""
    _DEPENDS[name]
    return LAB05_DIR / f"{name}.py"


def load(name: str) -> ModuleType:
    """Return 05's module `name`, executing it at most once per process.

    Dependencies load first so the module's own bare imports resolve. An
    entry already in sys.modules is reused only when it was loaded from 05's
    file; anything else under that name is left alone and shadowed.
    """
    for dependency in _DEPENDS[name]:
        load(dependency)
    path = path_for(name)
    existing = sys.modules.get(name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and Path(existing_file).resolve() == path:
            return existing
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Never leave a half-initialised module registered under 05's name.
        sys.modules.pop(name, None)
        raise
    return module


def sibling(name: str) -> ModuleType:
    """Return module 06's own script `name`, executing it at most once per process.

    Registered as lab06_<name>, so it can never shadow a 05 module or an
    installed package. A missing file is an ImportError naming the path.
    """
    module_name = f"{SIBLING_PREFIX}{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = LAB06_DIR / f"{name}.py"
    if not path.exists():
        raise ImportError(f"module 06 has no script {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
