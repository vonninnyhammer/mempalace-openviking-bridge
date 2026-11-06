#!/usr/bin/env python3
"""Score MemPalace drawers and write the `importance` metadata L1 sorts on.

`layers.py` ranks L1 ("essential story") by an `importance` field, then by
`filed_at`. Nothing in the ingest pipeline ever writes `importance`, so every
drawer ties at the default 3.0 and L1 degenerates into "newest first, hard
truncated at 197 chars" — which is how mid-sentence fragments end up being the
first thing loaded into a session.

This fills in that field with two components:

  base     — what kind of content it is. A chunk carrying a user turn is the
             thing the user actually said and outranks a chunk that is only
             internal reasoning, which in turn outranks tool-call markers.
  decay    — Ebbinghaus-style exponential forgetting applied to the age of the
             *content* (the session's start date, not the filing date), so a
             month-old decision fades below a week-old one. This is ranking
             only: nothing is deleted, and re-running after a fresh mine
             restores any drawer whose score should rise again.

Idempotent. Safe to run on every capture cycle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / ".mempalace-venv/lib/python3.10/site-packages"))

import chromadb  # noqa: E402

PALACE = Path.home() / ".mempalace/palace"
INDEX = Path.home() / "mempalace-viking/ingest/opencode/sessions-index.meta.json"
COLLECTION = "mempalace_drawers"

# Half-life in days for the decay term. ~30 days keeps a month-old decision
# clearly below a fresh one without erasing it from the top-N entirely.
HALF_LIFE_DAYS = 45.0
DECAY_FLOOR = 0.25

DECISION_CUES = re.compile(
    r"\b(decided|decision|we will|we're going|confirmed|agreed|conclusion|"
    r"the plan is|settled on|final answer|approved|resolved|standing rule|"
    r"from now on|going forward)\b",
    re.IGNORECASE,
)
# Names that mark a chunk as being about your own projects or machines. Set
# MEMPALACE_PROPER_NOUNS to a comma-separated list to use your own.
_NOUNS = [n.strip() for n in os.environ.get("MEMPALACE_PROPER_NOUNS", "OpenViking,MemPalace").split(",") if n.strip()]
PROPER_NOUN = re.compile(r"\b(" + "|".join(map(re.escape, _NOUNS)) + r")\b")
FILE_MANIFEST = "## Files written or edited by OpenCode"


def base_score(doc: str) -> float:
    lines = doc.splitlines()
    has_user = any(l.startswith("> ") for l in lines)
    has_assistant = any(
        l.strip() and not l.startswith("> ") and not l.startswith("[") and not l.startswith("#")
        for l in lines
    )
    has_reasoning = any(l.startswith("[reasoning]") for l in lines)
    has_tool = any(l.startswith("[tool:") for l in lines)

    if has_user:
        score = 5.0
    elif has_assistant:
        score = 3.5
    elif has_reasoning:
        score = 1.5
    elif has_tool:
        score = 0.5
    else:
        score = 1.0

    if DECISION_CUES.search(doc):
        score += 1.5
    if PROPER_NOUN.search(doc):
        score += 0.5
    if FILE_MANIFEST in doc:
        score += 0.5
    return score


def content_age_days(doc: str, meta: dict, index: dict) -> float | None:
    src = (meta or {}).get("source_file") or ""
    m = re.search(r"opencode-(ses_[A-Za-z0-9]+)\.txt", src)
    if m and m.group(1) in index:
        stamp = index[m.group(1)].get("date")
        if stamp:
            try:
                when = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
            delta = datetime.now(timezone.utc) - when
            return max(0.0, delta.total_seconds() / 86400.0)
    # Fall back to the transcript header, then to filing time.
    m = re.search(r"^# started: (\d{4}-\d{2}-\d{2})", doc, re.MULTILINE)
    if m:
        try:
            when = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return max(0.0, (datetime.now(timezone.utc) - when).total_seconds() / 86400.0)
    filed = (meta or {}).get("filed_at")
    if isinstance(filed, str) and filed:
        try:
            when = datetime.fromisoformat(filed.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - when).total_seconds() / 86400.0)
        except ValueError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palace", type=Path, default=PALACE)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--wing", default=None, help="only rescore this wing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index: dict = {}
    if args.index.exists():
        index = json.loads(args.index.read_text())

    client = chromadb.PersistentClient(path=str(args.palace))
    col = client.get_collection(COLLECTION)

    total = col.count()
    batch = 500
    offset = 0
    changed = 0
    scanned = 0
    buckets: dict[str, int] = {}

    while True:
        kwargs = {"include": ["documents", "metadatas"], "limit": batch, "offset": offset}
        if args.wing:
            kwargs["where"] = {"wing": args.wing}
        res = col.get(**kwargs)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        ids = res.get("ids") or []
        if not docs:
            break

        updates = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            scanned += 1
            doc = doc or ""
            meta = dict(meta or {})
            b = base_score(doc)
            age = content_age_days(doc, meta, index)
            decay = DECAY_FLOOR
            if age is not None:
                decay = max(DECAY_FLOOR, math.pow(0.5, age / HALF_LIFE_DAYS))
            score = round(b * decay, 4)
            band = "high" if score >= 4 else "mid" if score >= 2 else "low"
            buckets[band] = buckets.get(band, 0) + 1
            if meta.get("importance") != score:
                meta["importance"] = score
                meta["base_score"] = b
                meta["content_age_days"] = None if age is None else round(age, 2)
                updates.append((doc_id, meta))
                changed += 1

        if updates and not args.dry_run:
            col.update(ids=[u[0] for u in updates], metadatas=[u[1] for u in updates])

        offset += len(docs)
        if len(docs) < batch or offset >= total:
            break

    verb = "would rescore" if args.dry_run else "rescored"
    print(
        f"scanned={scanned} {verb}={changed} "
        f"bands={buckets.get('high', 0)}high/{buckets.get('mid', 0)}mid/{buckets.get('low', 0)}low"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
