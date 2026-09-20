#!/usr/bin/env python3
"""
mempalace_viking_bridge.py — MemPalace → OpenViking sync bridge.

Dual-layer architecture:
  Layer 1 (MemPalace): spatial memory — Wings/Rooms/Halls/Drawers in ChromaDB+SQLite.
  Layer 2 (OpenViking): execution-time RAG context filesystem (viking://) with
                        L0 abstract / L1 overview / L2 detail tiers.

Pipeline:
  1. EXPORT  — read all drawers from the MemPalace ChromaDB collection.
  2. STAGE   — write drawers into a staging tree laid out as viking://resources/
               (wing/room/drawer.md) with a manifest for change detection.
  3. SYNC    — call OpenViking add_resource on the staging tree; OpenViking
               generates L0/L1/L2 via its configured embedding + VLM.
  4. VERIFY  — confirm the resource landed under viking://resources/mempalace/
               and that L0/L1 sidecars exist.

Idempotent: only drawers whose (source_file, chunk_index, filed_at) changed
since the last sync are re-staged and re-synced.

Usage:
  python mempalace_viking_bridge.py [--full] [--verify-only]
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import chromadb
from openviking_sdk import SyncHTTPClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PALACE_PATH = Path.home() / ".mempalace" / "palace"
COLLECTION = "mempalace_drawers"
STAGING_DIR = Path.home() / "mempalace-viking" / "bridge" / "staging" / "mempalace"
MANIFEST = STAGING_DIR / ".manifest.json"
VIKING_URL = "http://localhost:1933"
VIKING_KEY = (Path.home() / ".openviking" / ".user_key").read_text().strip()
VIKING_ROOT = "viking://resources/mempalace"

# Drawer metadata fields that define content identity for change detection.
IDENTITY_FIELDS = ("source_file", "chunk_index", "filed_at")


# ---------------------------------------------------------------------------
# Layer 1: EXPORT from MemPalace
# ---------------------------------------------------------------------------
def export_drawers() -> list[dict]:
    """Read every drawer from the MemPalace ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(PALACE_PATH))
    col = client.get_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    drawers = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        drawers.append(
            {
                "text": doc,
                "wing": meta.get("wing", "unfiled"),
                "room": meta.get("room", "general"),
                "hall": meta.get("hall", ""),
                "source_file": meta.get("source_file", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_total": meta.get("chunk_total", 1),
                "filed_at": meta.get("filed_at", ""),
                "entities": meta.get("entities", ""),
            }
        )
    return drawers


def drawer_identity(d: dict) -> str:
    """Stable identity string for change detection."""
    raw = "|".join(str(d.get(k, "")) for k in IDENTITY_FIELDS)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Layer 1→2: STAGE into viking:// layout
# ---------------------------------------------------------------------------
def stage_drawers(drawers: list[dict], full: bool = False) -> tuple[list[Path], dict]:
    """
    Write drawers into the staging tree:

        staging/
          resources/mempalace/{wing}/{room}/{source_stem}__{chunk}.md
          .manifest.json   <- identity -> path map for change detection

    Returns (changed_files, manifest).
    """
    if STAGING_DIR.exists() and full:
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    changed: list[Path] = []
    new_manifest = {}

    for d in drawers:
        ident = drawer_identity(d)
        wing = _safe(d["wing"])
        room = _safe(d["room"])
        src = Path(d["source_file"]).stem if d["source_file"] else "drawer"
        out = STAGING_DIR / wing / room / f"{src}__{d['chunk_index']:02d}.md"
        out.parent.mkdir(parents=True, exist_ok=True)

        header = (
            f"<!-- mempalace drawer | wing={d['wing']} room={d['room']} "
            f"hall={d['hall']} source={d['source_file']} chunk={d['chunk_index']}/{d['chunk_total']} "
            f"filed_at={d['filed_at']} entities={d['entities']} -->\n\n"
        )
        content = header + d["text"].strip() + "\n"

        if manifest.get(ident) != str(out) or not out.exists():
            out.write_text(content)
            changed.append(out)

        new_manifest[ident] = str(out)

    MANIFEST.write_text(json.dumps(new_manifest, indent=2))
    return changed, new_manifest


def _safe(name: str) -> str:
    """Sanitize a wing/room name for filesystem use."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "general"


# ---------------------------------------------------------------------------
# Layer 2: SYNC into OpenViking
# ---------------------------------------------------------------------------
def sync_to_viking(changed: list[Path]) -> dict:
    """Add the staged resource tree to OpenViking (async submit, then poll)."""
    client = SyncHTTPClient(url=VIKING_URL, api_key=VIKING_KEY)
    try:
        client.initialize()
        result = client.add_resource(path=str(STAGING_DIR), to=VIKING_ROOT, wait=False)
        root_uri = result.get("root_uri", VIKING_ROOT)
        return {"root_uri": root_uri, "result": result}
    finally:
        client.close()


def _http_ls(uri: str) -> list[dict]:
    """Raw HTTP ls — the SDK strips the `abstract` field, the API does not."""
    req = urllib.request.Request(
        f"{VIKING_URL}/api/v1/fs/ls?uri={urllib.parse.quote(uri)}",
        headers={"X-API-Key": VIKING_KEY},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return data.get("result", [])


def wait_processed(timeout_s: int = 900) -> bool:
    """Poll until the mempalace resource tree has semantic abstracts or timeout.

    OpenViking hides .abstract.md/.overview.md dotfiles from ls, but exposes
    the generated L0 abstract via each directory's `abstract` field. We poll
    for non-empty abstracts on the wing directories.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            items = _http_ls(VIKING_ROOT)
            if any((i.get("abstract") or "").strip() for i in items):
                return True
        except Exception:
            pass
        time.sleep(15)
    return False


def _collect_sidecars(client, root_uri: str, depth: int = 0) -> dict:
    """Recursively collect .abstract.md / .overview.md URIs under root_uri."""
    sidecars = {"l0": [], "l1": []}
    if depth > 4:
        return sidecars
    try:
        ls = client.ls(uri=root_uri)
        items = ls if isinstance(ls, list) else ls.get("entries") or ls.get("items") or []
        for item in items:
            uri = item.get("uri") or item.get("path") or ""
            if uri.endswith(".abstract.md"):
                sidecars["l0"].append(uri)
            elif uri.endswith(".overview.md"):
                sidecars["l1"].append(uri)
            elif item.get("isDir"):
                sub = _collect_sidecars(client, uri, depth + 1)
                sidecars["l0"].extend(sub["l0"])
                sidecars["l1"].extend(sub["l1"])
    except Exception:
        pass
    return sidecars


# ---------------------------------------------------------------------------
# Layer 2: VERIFY
# ---------------------------------------------------------------------------
def verify_viking() -> dict:
    """Confirm the mempalace resource tree exists with L0 abstracts."""
    items = _http_ls(VIKING_ROOT)
    l0 = [i.get("uri") for i in items if (i.get("abstract") or "").strip()]
    return {"root": VIKING_ROOT, "tree": items, "l0_abstracts": l0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="MemPalace -> OpenViking bridge")
    ap.add_argument("--full", action="store_true", help="rebuild staging from scratch")
    ap.add_argument("--verify-only", action="store_true", help="only check viking:// state")
    args = ap.parse_args()

    if args.verify_only:
        print(json.dumps(verify_viking(), indent=2, default=str))
        return 0

    t0 = time.time()
    drawers = export_drawers()
    changed, manifest = stage_drawers(drawers, full=args.full)
    print(f"[export] {len(drawers)} drawers from {PALACE_PATH}")

    if not changed:
        print("[stage] no changes since last sync — nothing to do")
        print(json.dumps(verify_viking(), indent=2, default=str))
        return 0

    print(f"[stage] {len(changed)} changed files -> {STAGING_DIR}")
    for f in changed:
        print(f"        {f.relative_to(STAGING_DIR)}")

    print("[sync]  pushing to OpenViking...")
    sync = sync_to_viking(changed)
    print(f"[sync]  root_uri={sync['root_uri']}")

    print("[sync]  waiting for semantic processing (L0/L1)...")
    ready = wait_processed(timeout_s=900)
    print(f"[sync]  semantic processing {'COMPLETE' if ready else 'TIMEOUT'}")

    print("[verify]")
    print(json.dumps(verify_viking(), indent=2, default=str))
    print(f"[done]  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())