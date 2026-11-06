#!/usr/bin/env python3
"""Export OpenCode sessions from opencode.db into MemPalace transcript format.

Reads the OpenCode SQLite store read-only and writes one plain-text transcript
per session, using MemPalace's `> user / assistant` exchange format so
`mempalace mine --mode convos` consumes them unchanged.

Incremental by default: a session is re-exported only when its rendered text
actually changes, which keeps mempalace's mtime-aware skip check effective and
makes the 10-minute timer nearly free. `--full` forces a re-render of
everything.

Tool *outputs* are deliberately excluded (bulk of the DB, mostly throwaway
noise). Tool *names* are kept as one-line markers so the reasoning around them
stays interpretable, and every file OpenCode wrote or edited is recorded in a
per-session footer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".local/share/opencode/opencode.db"
DEFAULT_OUT = Path.home() / "mempalace-viking/ingest/opencode"
DEFAULT_STATE = Path.home() / "mempalace-viking/ingest/opencode/.export-state.meta.json"

# Tools whose input names a file OpenCode created or modified. Recorded in the
# transcript footer so "what did I write about X" is searchable.
WRITE_TOOLS = {"write", "edit", "patch", "apply_patch", "multiedit"}

# Ceiling on the per-session file manifest, so a session that touched a
# generated tree cannot turn the footer into the bulk of the transcript.
MAX_FILES_LISTED = 200


def connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        sys.exit(f"opencode db not found: {db}")
    # Read-only, and never through a symlink. mode=ro still consults the -wal
    # sidecar, which is how we see writes that have not been checkpointed yet.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    return con


def iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


_SECRET_PATTERNS = (
    # OpenRouter keys, longest-first so the specific rule wins over the generic one.
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"), "[REDACTED:openrouter-key]"),
    # Generic provider keys (OpenAI and friends), not matching sk-or-v1-*.
    (re.compile(r"sk-(?!or-v1-)[A-Za-z0-9_-]{32,}"), "[REDACTED:api-key]"),
    # Common bearer/secret assignments, value only.
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._-]{20,}"), r"\1[REDACTED:token]"),
)


def redact(text: str) -> str:
    """Strip credentials so they never enter the durable palace corpus.

    Sessions are exported verbatim, so any key pasted into a conversation would
    otherwise be embedded, indexed, and recalled forever. Partial or elided
    references (sk-or-v1-cf7...475) are left alone - they are not usable keys
    and redacting them would destroy legitimate memory content.
    """
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def clean(text: str) -> str:
    return redact(text.replace("\x00", "�")).strip()


def tool_marker(part: dict) -> str | None:
    name = part.get("tool") or "tool"
    state = part.get("state") or {}
    status = state.get("status") or "?"
    if status == "error":
        return f"[tool: {name} — error]"
    return f"[tool: {name}]"


def write_targets(part: dict) -> list[str]:
    name = part.get("tool") or ""
    if name not in WRITE_TOOLS:
        return []
    args = (part.get("state") or {}).get("input") or {}
    out = []
    for key in ("filePath", "path", "file"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
    return out


def render(con: sqlite3.Connection, session: dict) -> tuple[str, int]:
    """Render one session to transcript text. Returns (text, last_activity_ms)."""
    sid = session["id"]
    parts_by_msg: dict[str, list[dict]] = {}
    for pid, data, tcreated in con.execute(
        "SELECT message_id, data, time_created FROM part WHERE session_id=? ORDER BY time_created",
        (sid,),
    ):
        try:
            j = json.loads(data)
        except (ValueError, TypeError):
            continue
        parts_by_msg.setdefault(pid, []).append(j)
        del tcreated

    lines: list[str] = []
    touched: list[str] = []
    last = session["time_created"]

    header = [
        f"# OpenCode session: {session.get('title') or '(untitled)'}",
        f"# id: {sid}",
        f"# started: {iso(session['time_created'])}",
        f"# updated: {iso(session['time_updated'])}",
        f"# cwd: {session.get('directory') or '/'}",
    ]
    if session.get("parent_id"):
        header.append(f"# subagent-of: {session['parent_id']}")
    if session.get("agent"):
        header.append(f"# agent: {session['agent']}")
    model = session.get("model")
    if isinstance(model, str) and model.strip():
        if model.strip().startswith("{"):
            try:
                parsed = json.loads(model)
                model = parsed.get("modelID") or parsed.get("id") or ""
                provider = parsed.get("providerID") or ""
                model = f"{provider}/{model}" if provider and model else model
            except ValueError:
                model = ""
        if model:
            header.append(f"# model: {model}")
    lines.extend(header)
    lines.append("")

    rows = con.execute(
        "SELECT id, data, time_created FROM message WHERE session_id=? ORDER BY time_created",
        (sid,),
    )
    for mid, mdata, tcreated in rows:
        try:
            m = json.loads(mdata)
        except (ValueError, TypeError):
            continue
        last = max(last, tcreated)
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        body: list[str] = []
        for part in parts_by_msg.get(mid, []):
            ptype = part.get("type")
            if ptype == "text":
                text = clean(part.get("text") or "")
                if text:
                    body.append(("USER" if role == "user" else "ASST", text))
            elif ptype == "reasoning":
                text = clean(part.get("text") or "")
                if text:
                    body.append(("REASON", f"[reasoning] {text}"))
            elif ptype == "tool":
                marker = tool_marker(part)
                if marker:
                    body.append(("TOOL", marker))
                touched.extend(write_targets(part))
            elif ptype == "compaction":
                body.append(("META", "[context compacted]"))
            # NB: `patch` parts are opencode's whole-tree snapshot manifest
            # (every tracked path), not a record of edits. Emitting it produced
            # a 63MB footer listing $HOME. Real writes are already captured from
            # the write/edit tool inputs above.

        if not body:
            continue
        # Within a message, lead with the human-visible exchange and push
        # internal reasoning and tool markers after it. MemPalace's L1 renders
        # a 197-char preview of each drawer, so whatever comes first is what a
        # session sees at wake-up; opencode's native order (reasoning, tools,
        # then reply) put mid-thought fragments in that slot. Nothing is
        # dropped, only reordered.
        rank = {"USER": 0, "ASST": 1, "REASON": 2, "TOOL": 3, "META": 4}
        body.sort(key=lambda kv: rank.get(kv[0], 5))
        for kind, text in body:
            if kind == "USER":
                lines.append(f"> {text}")
                lines.append("")
            elif kind == "ASST":
                lines.append(text)
                lines.append("")
            else:
                lines.append(text)
                lines.append("")

    if touched:
        uniq = sorted(set(touched))
        lines.append("## Files written or edited by OpenCode in this session")
        for path in uniq[:MAX_FILES_LISTED]:
            lines.append(f"- {path}")
        if len(uniq) > MAX_FILES_LISTED:
            lines.append(f"- ... and {len(uniq) - MAX_FILES_LISTED} more")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", last


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return {"exported": {}, "last_activity": 0}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--full", action="store_true", help="re-render every session")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    con = connect(args.db)
    state = load_state(args.state)
    exported: dict = {} if args.full else dict(state.get("exported") or {})

    sessions = [
        {
            "id": r[0],
            "title": r[1],
            "directory": r[2],
            "parent_id": r[3],
            "agent": r[4],
            "model": r[5],
            "time_created": r[6],
            "time_updated": r[7],
        }
        for r in con.execute(
            "SELECT id, title, directory, parent_id, agent, model, time_created, time_updated"
            " FROM session ORDER BY time_created"
        )
    ]

    written = skipped = failed = 0
    newest = int(state.get("last_activity") or 0)

    for sess in sessions:
        sid = sess["id"]
        try:
            text, last = render(con, sess)
        except sqlite3.Error as exc:
            print(f"  ! {sid}: {exc}", file=sys.stderr)
            failed += 1
            continue

        digest = hashlib.sha256(text.encode()).hexdigest()
        target = out / f"opencode-{sid}.txt"

        if digest == exported.get(sid) and target.exists() and not args.full:
            skipped += 1
            newest = max(newest, last)
            continue

        if not args.dry_run:
            tmp = target.with_suffix(".tmp")
            tmp.write_text(text)
            os.replace(tmp, target)
        exported[sid] = digest
        written += 1
        newest = max(newest, last)

    if not args.dry_run:
        # Forget sessions that no longer exist in the db so state cannot grow
        # without bound, and drop their transcripts.
        live = {s["id"] for s in sessions}
        for gone in set(exported) - live:
            stale = out / f"opencode-{gone}.txt"
            if stale.exists():
                stale.unlink()
            exported.pop(gone, None)
        state["exported"] = exported
        state["last_activity"] = newest
        state["last_run"] = iso(int(dt.datetime.now().timestamp() * 1000))
        save_state(args.state, state)

        # Session-date index. Filenames carry only the session id, so the
        # ranker needs this to age a drawer by when its content happened
        # rather than by when it was filed.
        index = {
            s["id"]: {
                "date": iso(s["time_created"]),
                "updated": iso(s["time_updated"]),
                "title": s.get("title") or "",
                "cwd": s.get("directory") or "",
            }
            for s in sessions
        }
        idx = out / "sessions-index.meta.json"
        tmp = idx.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
        os.replace(tmp, idx)

    print(
        f"sessions={len(sessions)} written={written} unchanged={skipped} "
        f"failed={failed} out={out}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
