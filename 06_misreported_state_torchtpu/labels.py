"""
Module 6: The words for where a record's labels came from.

A judge run never reads labels: 05's prompts render the goal, the boundaries,
the steps and the final report, and nothing else. Labels matter only when a
run is scored, and a score means nothing unless the reader knows whose
labels it was scored against. So every table and every metadata block 06
writes says which labels were used, in these words and no others:

    student-adjudicated
        The record went through adjudicate.py and a person promoted it. In
        the student track that person is you, and the adjudication note on
        the record carries your name and the date.
    template-construction labels, not adjudicated
        A staged tier-2 record. Its label is whatever 05's generator planted
        when it built the trajectory from a template. Nobody has read it.
    proposed labels, not adjudicated
        Any other staged record nobody has decided, such as a tamper draft
        before adjudication.
    rejected at adjudication, no label
        The reviewer pressed r. The record has no ground truth, so nothing
        is scored against it.
    adjudicated
        Corpus mode, the author track: the loader refuses any record that
        is not adjudicated, so the whole file is.

The mode is the loader's, not a guess. A file read with --staged or
--allow-staged is in staged mode, and an adjudicated record in it is one the
person who built that file promoted. A file read without those flags is a
scored corpus.

Why scoring_labels() exists. 05's score.py --allow-staged was written for
files where nothing is adjudicated, and reads `record.labels or
record.proposed_labels`. The student eval set mixes the two kinds. A record
a reviewer adjudicated as clean has an empty labels list, so in a mixed file
that line would swap the reviewer's decision for the proposal it overruled.
scoring_labels() keys on `adjudicated`, never on whether the list is empty.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import lab05

_schema = lab05.load("schema")

LABELS_STUDENT = "student-adjudicated"
LABELS_TEMPLATE = "template-construction labels, not adjudicated"
LABELS_PROPOSED = "proposed labels, not adjudicated"
LABELS_REJECTED = "rejected at adjudication, no label"
LABELS_CORPUS = "adjudicated"
LABEL_SOURCES: tuple[str, ...] = (LABELS_STUDENT, LABELS_TEMPLATE, LABELS_PROPOSED, LABELS_REJECTED, LABELS_CORPUS)

# 05's generator stamps this on every tier-2 record it builds from a template.
TEMPLATE_SOURCE = "tier2_template"
# adjudicate.py writes "<name> <date> rejected: <reason>" on a rejected record.
REJECTED_MARK = " rejected: "
JUDGE_NOTE = "the judge prompt renders no labels; this block says what the judged records carried, for scoring"


def is_rejected(record: Any) -> bool:
    return not record.adjudicated and REJECTED_MARK in f" {record.adjudication_note}"


def label_source(record: Any, staged: bool) -> str:
    """Which of the five phrases describes this record's labels."""
    if not staged:
        return LABELS_CORPUS
    if record.adjudicated:
        return LABELS_STUDENT
    if is_rejected(record):
        return LABELS_REJECTED
    if record.source == TEMPLATE_SOURCE:
        return LABELS_TEMPLATE
    return LABELS_PROPOSED


def scoring_labels(record: Any) -> list | None:
    """The labels a scorer may hold this record to, or None when it has no ground truth.

    An adjudicated record answers with its labels even when the list is
    empty: clean is a decision. A rejected record answers None. Anything
    else answers with its proposal.
    """
    if record.adjudicated:
        return list(record.labels)
    if is_rejected(record):
        return None
    return list(record.proposed_labels)


def adjudicator(record: Any) -> str | None:
    """The name adjudicate.py stamped first in the note, for an adjudicated record."""
    if not record.adjudicated:
        return None
    words = record.adjudication_note.split()
    return words[0] if words else None


def _tier_text(sources: dict[str, int], adjudicators: list[str]) -> str:
    def one(source: str) -> str:
        if source == LABELS_STUDENT and adjudicators:
            return f"{source} (by {', '.join(adjudicators)})"
        return source

    if len(sources) == 1:
        return one(next(iter(sources)))
    return " + ".join(f"{count} {one(source)}" for source, count in sources.items())


def summarize(records: list, staged: bool) -> dict[str, Any]:
    """The label-source block for a set of records, split by tier the way 05's scorer splits its tables."""
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in _schema.TIERS:
        members = [record for record in records if record.tier == tier]
        if not members:
            continue
        counts = Counter(label_source(record, staged) for record in members)
        sources = {source: counts[source] for source in LABEL_SOURCES if counts[source]}
        names = sorted({name for record in members if (name := adjudicator(record))}) if staged else []
        by_tier[tier] = {
            "records": len(members),
            "sources": sources,
            "adjudicators": names,
            "labels": _tier_text(sources, names),
        }
    statement = "; ".join(
        f"tier {tier}, {entry['records']} records: {entry['labels']}" for tier, entry in by_tier.items()
    )
    return {
        "mode": "staged" if staged else "corpus",
        "read_by_judge": False,
        "note": JUDGE_NOTE,
        "by_tier": by_tier,
        "statement": statement or "no records",
    }


def tier_labels(summary: dict[str, Any], tier: str) -> str:
    """The phrase for one tier's table title."""
    return summary["by_tier"].get(tier, {}).get("labels", "no records")
