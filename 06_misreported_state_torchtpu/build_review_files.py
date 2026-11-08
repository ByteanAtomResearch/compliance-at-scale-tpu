"""
Module 6: Build the Part 3 review files from the staged tier-2 batch.

The human review for Part 3 covers 60 tier-2 positives (10 per state, 7 in
full fidelity and 3 in the reduced region) plus 14 authorization_bypass
policy pairs that top up the policy_withheld test. adjudicate.py walks every
pending record in the file it is given, so these records get their own
staged files inside 06. sample_data/staging/ is never modified and 05 stays
frozen.

The id lists live here as literals, not as a seeded draw, so the cut is a
reviewable artifact: anyone can see which records were chosen and rebuild
the two files byte for byte. Lines are copied from the source unchanged and
in source order. --check rebuilds in memory and compares against the
committed files without writing anything.

Usage:
    python 06_misreported_state_torchtpu/build_review_files.py
    python 06_misreported_state_torchtpu/build_review_files.py --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "sample_data" / "staging" / "candidates_tier2_batch1.jsonl"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "staging"

CUT60_FILENAME = "candidates_tier2_cut60.jsonl"
TOPUP14_FILENAME = "candidates_tier2_topup14.jsonl"

# Seeded draw (seed 20260812, adjudicate.py's default) recorded by hand in the
# module 06 spec: 7 full-fidelity and 3 reduced-region records per state.
# authorization_bypass keeps three policy_visible / policy_withheld pairs whole.
CUT60_IDS: tuple[str, ...] = tuple(
    f"traj-t2-{short}"
    for short in (
        # scope_violation
        "sv-002 sv-003 sv-013 sv-018 sv-026 sv-027 sv-034 sv-016 sv-020 sv-036 "
        # specification_gaming
        "sg-040 sg-045 sg-059 sg-060 sg-064 sg-068 sg-069 sg-050 sg-058 sg-070 "
        # sensitive_state_exposure
        "sse-078 sse-082 sse-085 sse-087 sse-089 sse-102 sse-106 sse-088 sse-092 sse-108 "
        # unsafe_continuation
        "uc-117 uc-136 uc-139 uc-141 uc-144 uc-147 uc-152 uc-126 uc-134 uc-142 "
        # misreported_state
        "mr-158 mr-166 mr-167 mr-170 mr-178 mr-179 mr-187 mr-160 mr-172 mr-188 "
        # authorization_bypass
        "ab-191-pv ab-191-pw ab-205-pv ab-205-pw ab-212 ab-216 ab-224 ab-194-pv ab-194-pw ab-225"
    ).split()
)

# Seven more policy pairs, kept whole so the visible side matches the withheld
# side. 202 and 206 sit in the reduced region.
TOPUP14_PAIRS: tuple[str, ...] = ("192", "193", "197", "200", "204", "202", "206")
TOPUP14_IDS: tuple[str, ...] = tuple(
    f"traj-t2-ab-{pair}-{variant}" for pair in TOPUP14_PAIRS for variant in ("pv", "pw")
)

if len(CUT60_IDS) != 60 or len(set(CUT60_IDS)) != 60:
    raise ValueError(f"CUT60_IDS must hold 60 distinct ids, has {len(set(CUT60_IDS))}")
if len(TOPUP14_IDS) != 14 or len(set(TOPUP14_IDS)) != 14:
    raise ValueError(f"TOPUP14_IDS must hold 14 distinct ids, has {len(set(TOPUP14_IDS))}")
if set(CUT60_IDS) & set(TOPUP14_IDS):
    raise ValueError(f"CUT60_IDS and TOPUP14_IDS overlap: {sorted(set(CUT60_IDS) & set(TOPUP14_IDS))}")

REVIEW_FILES: dict[str, tuple[str, ...]] = {CUT60_FILENAME: CUT60_IDS, TOPUP14_FILENAME: TOPUP14_IDS}


def select_lines(source: str | Path, ids: tuple[str, ...]) -> list[str]:
    """Return the source lines whose record id is in ids, unchanged and in source order.

    Lines are kept byte for byte (including the trailing newline) so the
    review file is a strict subset of the staged batch, never a re-encoding.
    """
    wanted = set(ids)
    found: dict[str, str] = {}
    with open(source, encoding="utf-8", newline="") as f:
        for line in f:
            if not line.strip():
                continue
            record_id = json.loads(line)["id"]
            if record_id in wanted:
                found[record_id] = line if line.endswith("\n") else line + "\n"
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"{source}: {len(missing)} requested ids not found: {missing}")
    return list(found.values())


def build(source: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Write every review file and return line counts by filename."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for filename, ids in REVIEW_FILES.items():
        lines = select_lines(source, ids)
        with open(out_dir / filename, "w", encoding="utf-8", newline="") as f:
            f.write("".join(lines))
        counts[filename] = len(lines)
    return counts


def check(source: str | Path, out_dir: str | Path) -> list[str]:
    """Rebuild in memory and report every committed file that differs. Writes nothing."""
    problems: list[str] = []
    for filename, ids in REVIEW_FILES.items():
        expected = "".join(select_lines(source, ids)).encode("utf-8")
        path = Path(out_dir) / filename
        if not path.exists():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != expected:
            problems.append(f"{path}: differs from a rebuild out of {source}")
    return problems


def main(args: argparse.Namespace) -> None:
    if args.check:
        problems = check(args.source, args.out_dir)
        if problems:
            raise SystemExit("review files drifted from the staged batch:\n  " + "\n  ".join(problems))
        console.print(f"[green]Review files match a rebuild from {args.source}.[/green]")
        return
    counts = build(args.source, args.out_dir)
    for filename, count in counts.items():
        console.print(f"  {Path(args.out_dir) / filename}: {count} records")
    console.print("[green]Review files written. sample_data/staging/ was not modified.[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build module 06's staged review files from the tier-2 batch")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Staged tier-2 batch to cut from")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where the review files go")
    parser.add_argument("--check", action="store_true", help="Compare committed files to a rebuild; write nothing")
    main(parser.parse_args())
