"""
Module 6: Bridge from judge verdicts to the rai-checklist-cli report shapes.

Module 04 feeds rai-checklist-cli two shapes: a Markdown checklist and a
YAML `rai_evaluation` mapping of section title to status, flagged_count and
flagged_records. 04's generate_markdown_report and generate_yaml_report
loop over 04's three hardcoded heuristics, so they would report nothing for
the six trajectory states. bridge.py reproduces their shape for the six
states in 05's VIOLATION_STATES order, without calling or changing 04.

Two things are kept apart on purpose. Every verdict goes through 05's
schema.validate_verdict before anything is written, so an unknown state
cannot reach a compliance artifact nobody can audit. Recommended
enforcement is operator policy and never model output (05's verdict_to_dict
says so in the verdict's policy object), so it lives in its own policy
block and never inside a section's findings. A provenance block carries the
run's model, backend, prompt hash, parse errors and commit so an auditor
can trace the report back to a run log. It also carries the run log's
statement of where the judged records' labels came from (labels.py). The
report itself shows judge flags only and reads no label; the line is there
so a reader comparing the report with a scored table knows whose labels
that table used. A verdict the schema rejects makes the bridge exit 2 with
nothing written; any other error exits 1.

Usage:
    python 06_misreported_state_torchtpu/bridge.py \
        --verdicts results/06_verdicts_torchtpu.json \
        --markdown results/06_report.md --yaml results/06_report.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import lab05
import yaml
from rich.console import Console
from rich.table import Table

_schema = lab05.load("schema")

console = Console()

# Exit statuses. A refusal (a verdict 05's schema rejects) exits 2 so a script can tell it from any
# other error, which exits 1 the way 05's tools do. Six lines, duplicated in
# parity.py and bridge.py on purpose: a shared module would need the lab05
# registration dance in every test for six lines.
EXIT_REFUSED = 2


class Refusal(SystemExit):
    """The inputs were refused before anything was written. Carries the message for tests."""

    def __init__(self, message: str):
        super().__init__(EXIT_REFUSED)
        self.message = message


def refuse(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise Refusal(message)


UNRECORDED = "unrecorded"
PROVENANCE_FIELDS: tuple[str, ...] = (
    "model",
    "backend",
    "dtype",
    "prompt_version",
    "prompt_hash",
    "call_mode",
    "parse_errors",
    "model_revision",
    "container_image_digest",
)
POLICY_SENTENCE = "Recommended enforcement is operator policy applied after the verdict, never model output."
LABELS_SENTENCE = "This report shows what the judge flagged. No label was read to build it."


def title_for(state: str) -> str:
    """04's section title rule, applied to 05's state names."""
    return state.replace("_", " ").title()


def load_verdict_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if "verdicts" not in doc:
        raise SystemExit(f"{path}: no 'verdicts' key; the bridge needs an --eval-set run")
    return doc


def validate_verdicts(verdicts: dict[str, dict[str, Any]]) -> None:
    """05's own rule, applied before anything is written. The first failure exits 2."""
    for record_id, verdict in verdicts.items():
        try:
            _schema.validate_verdict(_schema.verdict_from_dict(verdict))
        except (ValueError, TypeError, KeyError) as exc:
            refuse(f"verdict for {record_id} is invalid: {exc}")


def flagged_by_state(verdicts: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    flagged: dict[str, list[str]] = {state: [] for state in _schema.VIOLATION_STATES}
    for record_id, verdict in verdicts.items():
        for state in {violation["state"] for violation in verdict.get("violations", [])}:
            flagged[state].append(record_id)
    return {state: sorted(ids) for state, ids in flagged.items()}


def section_rows(verdicts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """04's rai_evaluation shape: FAIL when any record flagged the state, PASS otherwise."""
    return {
        title_for(state): {
            "status": "FAIL" if ids else "PASS",
            "flagged_count": len(ids),
            "flagged_records": ids,
        }
        for state, ids in flagged_by_state(verdicts).items()
    }


def policy_block(verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Enforcement per record, sourced from each verdict's policy object, kept out of the sections."""
    sources = sorted(
        {
            verdict.get("policy", {}).get("source")
            for verdict in verdicts.values()
            if verdict.get("policy", {}).get("source")
        }
    )
    return {
        "source": "; ".join(sources) if sources else UNRECORDED,
        "recommended_enforcement": {
            record_id: _schema.verdict_from_dict(verdict).recommended_enforcement
            for record_id, verdict in verdicts.items()
        },
    }


def provenance_block(metadata: dict[str, Any]) -> dict[str, Any]:
    """Enough of the run log to find it again. Missing fields say so instead of disappearing."""
    block = {field: metadata.get(field, UNRECORDED) for field in PROVENANCE_FIELDS}
    git = metadata.get("git")
    block["git_sha"] = git.get("sha", UNRECORDED) if isinstance(git, dict) else UNRECORDED
    block["label_sources"] = label_statement(metadata)
    return block


def label_statement(metadata: dict[str, Any]) -> str:
    """The run log's own words for whose labels the judged records carried. 05's runner records none."""
    sources = metadata.get("label_sources")
    return sources.get("statement", UNRECORDED) if isinstance(sources, dict) else UNRECORDED


def render_yaml(metadata: dict[str, Any], verdicts: dict[str, dict[str, Any]]) -> str:
    report = {
        "rai_evaluation": section_rows(verdicts),
        "policy": policy_block(verdicts),
        "provenance": provenance_block(metadata),
    }
    return yaml.dump(report, sort_keys=False, default_flow_style=False)


def render_markdown(metadata: dict[str, Any], verdicts: dict[str, dict[str, Any]]) -> str:
    total = len(verdicts)
    provenance = provenance_block(metadata)
    lines = [
        "# RAI Compliance Evaluation Report",
        "",
        f"**Model:** {provenance['model']}",
        f"**Backend:** {provenance['backend']}",
        f"**Records evaluated:** {total}",
        "",
    ]
    for state, ids in flagged_by_state(verdicts).items():
        clean = total - len(ids)
        lines.append(f"## {title_for(state)}")
        lines.append("")
        if not ids:
            lines.append(f"- [x] All records passed ({clean} clean, 0 flagged)")
        else:
            lines.append(f"- [ ] Issues found: {len(ids)} flagged, {clean} clean")
            lines.append(f"  - Flagged: {', '.join(ids)}")
        lines.append("")
    policy = policy_block(verdicts)
    lines += ["---", "", "## Policy", "", POLICY_SENTENCE, f"Source: {policy['source']}", ""]
    for record_id, action in policy["recommended_enforcement"].items():
        lines.append(f"- {record_id}: {action}")
    lines += ["", "## Provenance", ""]
    for key, value in provenance.items():
        lines.append(f"- {key}: {value}")
    lines += ["", LABELS_SENTENCE, ""]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    if not args.markdown and not args.yaml:
        raise SystemExit("pass --markdown, --yaml, or both")
    doc = load_verdict_file(args.verdicts)
    verdicts = doc["verdicts"]
    metadata = doc.get("metadata", {})
    validate_verdicts(verdicts)
    rendered = ((args.markdown, render_markdown(metadata, verdicts)), (args.yaml, render_yaml(metadata, verdicts)))
    for path, text in rendered:
        if path:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            console.print(f"[green]Wrote {out_path}[/green]")
    table = Table(title=f"rai_evaluation sections for {args.verdicts}")
    for column in ("Section", "Status", "Flagged"):
        table.add_column(column)
    for title, row in section_rows(verdicts).items():
        table.add_row(title, row["status"], str(row["flagged_count"]))
    console.print(table)
    console.print(
        f"Labels on the judged records: {label_statement(metadata)}. {LABELS_SENTENCE}",
        highlight=False,
        soft_wrap=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 6: verdicts to rai-checklist-cli report shapes")
    parser.add_argument("--verdicts", required=True, help="Verdict file from 05's runner or 06's run.py")
    parser.add_argument("--markdown", default=None, help="Markdown checklist path")
    parser.add_argument("--yaml", default=None, help="YAML rai_evaluation path")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
