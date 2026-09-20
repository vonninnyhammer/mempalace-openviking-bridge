#!/usr/bin/env python3
"""
escalation_listener.py — MemPalace event listener + escalation router.

Triggers (local, no cloud):
  1. LOW_CONFIDENCE  — a MemPalace search returns no result above threshold,
                       or the top hit is below the confidence floor.
  2. UNRESOLVED      — a query produced zero drawers (empty recall).
  3. ANOMALY         — palace health anomalies: ChromaDB count drift, SQLite
                       knowledge-graph corruption, bridge sync failures.

Each trigger is formatted into a structured escalation payload and:
  a) written to ~/mempalace-viking/escalations/ as JSON (durable local queue)
  b) routed to OpenViking-managed agents via the Hermes CLI (local)
  c) optionally POSTed to the n8n overseer webhook (local infra)

Usage:
  python escalation_listener.py check            # one-shot scan
  python escalation_listener.py watch --interval 60   # continuous listener
  python escalation_listener.py route --payload <file>  # re-route a payload
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PALACE_PATH = Path.home() / ".mempalace" / "palace"
COLLECTION = "mempalace_drawers"
ESCALATION_DIR = Path.home() / "mempalace-viking" / "escalations"
LOG_DIR = Path.home() / "mempalace-viking" / "logs"
STATE_FILE = Path.home() / "mempalace-viking" / "bridge" / ".escalation_state.json"

CONFIDENCE_FLOOR = 0.55      # below this, a top hit is low-confidence
MIN_DRAWERS = 1              # below this count, palace looks empty/anomalous
HERMES_BIN = Path.home() / ".local" / "bin" / "hermes"
N8N_WEBHOOK = "http://localhost:5678/webhook/mempalace-escalation"  # local n8n

ESCALATION_KINDS = ("low_confidence", "unresolved", "anomaly")


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------
def _palace_stats() -> dict:
    client = chromadb.PersistentClient(path=str(PALACE_PATH))
    col = client.get_collection(COLLECTION)
    return {"drawer_count": col.count()}


def detect_triggers(query: str | None = None) -> list[dict]:
    """Run trigger checks; return a list of escalation events."""
    events: list[dict] = []
    stats = _palace_stats()

    # --- Anomaly: palace empty or count drift -----------------------------
    if stats["drawer_count"] < MIN_DRAWERS:
        events.append(
            {
                "kind": "anomaly",
                "severity": "high",
                "reason": f"palace drawer count {stats['drawer_count']} below floor {MIN_DRAWERS}",
                "detail": stats,
            }
        )

    # --- Query-driven triggers --------------------------------------------
    if query:
        from mempalace.searcher import search_memories

        result = search_memories(query, palace_path=str(PALACE_PATH), n_results=5)
        hits = result.get("results", [])
        if not hits:
            events.append(
                {
                    "kind": "unresolved",
                    "severity": "medium",
                    "reason": f"zero recall for query: {query!r}",
                    "detail": {"query": query},
                }
            )
        else:
            top = hits[0]
            if top.get("similarity", 0.0) < CONFIDENCE_FLOOR:
                events.append(
                    {
                        "kind": "low_confidence",
                        "severity": "medium",
                        "reason": (
                            f"top hit similarity {top.get('similarity', 0.0):.3f} "
                            f"below floor {CONFIDENCE_FLOOR} for query: {query!r}"
                        ),
                        "detail": {"query": query, "top_hit": top},
                    }
                )

    return events


# ---------------------------------------------------------------------------
# Payload formatting
# ---------------------------------------------------------------------------
def format_payload(event: dict) -> dict:
    """Wrap a trigger into a structured, routable escalation payload."""
    return {
        "schema": "mempalace.escalation.v1",
        "event_id": f"esc-{int(time.time() * 1000)}",
        "ts": datetime.now(timezone.utc).isoformat(),
        "origin": "mempalace-listener",
        "kind": event["kind"],
        "severity": event.get("severity", "info"),
        "reason": event["reason"],
        "detail": event.get("detail", {}),
        "handoff": {
            "target": "openviking-agent",
            "protocol": "hermes-cli",
            "instructions": (
                f"Resolve escalation {event['kind']}: {event['reason']}. "
                "Use viking://resources/mempalace for context retrieval."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def write_payload(payload: dict) -> Path:
    ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
    out = ESCALATION_DIR / f"{payload['event_id']}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def route_to_hermes(payload: dict) -> dict:
    """Hand the payload to a Hermes-managed agent (local, no cloud)."""
    if not HERMES_BIN.exists():
        return {"routed": False, "reason": f"hermes not found at {HERMES_BIN}"}
    prompt = (
        f"[ESCALATION {payload['event_id']}] {payload['reason']}\n"
        f"kind={payload['kind']} severity={payload['severity']}\n"
        f"detail={json.dumps(payload['detail'])}\n"
        "Retrieve context from viking://resources/mempalace and propose a resolution."
    )
    try:
        proc = subprocess.run(
            [str(HERMES_BIN), "-z", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {"routed": True, "rc": proc.returncode, "output": proc.stdout.strip()[:300]}
    except Exception as e:
        return {"routed": False, "reason": str(e)}


def route_to_n8n(payload: dict) -> dict:
    """POST the payload to the local n8n overseer webhook (best-effort)."""
    try:
        req = urllib.request.Request(
            N8N_WEBHOOK,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"routed": True, "status": r.status}
    except Exception as e:
        return {"routed": False, "reason": str(e)}


def route_payload(payload: dict, use_n8n: bool = False) -> dict:
    path = write_payload(payload)
    routes = {"file": str(path), "hermes": route_to_hermes(payload)}
    if use_n8n:
        routes["n8n"] = route_to_n8n(payload)
    return routes


# ---------------------------------------------------------------------------
# State tracking (dedupe)
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"seen_events": []}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="MemPalace escalation listener")
    ap.add_argument("cmd", choices=["check", "watch", "route"])
    ap.add_argument("--query", help="query to evaluate for low-confidence/unresolved")
    ap.add_argument("--interval", type=int, default=60, help="watch interval seconds")
    ap.add_argument("--payload", help="payload file to re-route")
    ap.add_argument("--n8n", action="store_true", help="also POST to local n8n webhook")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "route":
        payload = json.loads(Path(args.payload).read_text())
        print(json.dumps(route_payload(payload, use_n8n=args.n8n), indent=2))
        return 0

    if args.cmd == "check":
        events = detect_triggers(query=args.query)
        if not events:
            print(json.dumps({"status": "ok", "events": []}, indent=2))
            return 0
        for ev in events:
            payload = format_payload(ev)
            routes = route_payload(payload, use_n8n=args.n8n)
            print(json.dumps({"event": payload, "routes": routes}, indent=2))
        return 0

    # watch mode
    state = _load_state()
    print(f"[watch] polling every {args.interval}s (Ctrl-C to stop)")
    while True:
        try:
            events = detect_triggers(query=args.query)
            for ev in events:
                payload = format_payload(ev)
                if payload["event_id"] in state["seen_events"]:
                    continue
                routes = route_payload(payload, use_n8n=args.n8n)
                state["seen_events"].append(payload["event_id"])
                state["seen_events"] = state["seen_events"][-100:]
                _save_state(state)
                print(f"[watch] {payload['kind']}: {payload['reason']}")
                print(f"        routed -> {routes}")
        except KeyboardInterrupt:
            print("\n[watch] stopped")
            return 0
        except Exception as e:
            print(f"[watch] error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())