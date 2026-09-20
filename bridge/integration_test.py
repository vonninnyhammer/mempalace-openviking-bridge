#!/usr/bin/env python3
"""
integration_test.py — end-to-end validation of the dual-layer architecture.

Flow:
  1. Generate a fresh sample log file.
  2. Ingest through MemPalace (mine -> drawers in ChromaDB/SQLite).
  3. Run the bridge (stage -> OpenViking add_resource -> L0/L1/L2).
  4. Verify an agent can retrieve the content via viking://.
  5. Assert zero cloud telemetry (all endpoints local/Tailscale/WireGuard).

Exit code 0 = PASS, 1 = FAIL.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VENV_PY = Path.home() / ".mempalace-venv" / "bin" / "python"
BRIDGE = Path.home() / "mempalace-viking" / "bridge" / "mempalace_viking_bridge.py"
INGEST_DIR = Path.home() / "mempalace-viking" / "ingest"
VIKING_URL = "http://localhost:1933"
USER_KEY = (Path.home() / ".openviking" / ".user_key").read_text().strip()

# Local-only network ranges (zero cloud telemetry).
LOCAL_NETS = ("127.", "10.", "192.168.", "100.86.", "fd7a:", "fe80:", "::1")


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    results: list[bool] = []
    print("=" * 60)
    print("DUAL-LAYER INTEGRATION TEST")
    print("=" * 60)

    # --- 1. Generate sample log -------------------------------------------
    print("\n[1/5] Generating sample log file")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = INGEST_DIR / f"pipeline-run-{ts}.txt"
    log.write_text(
        f"""# Pipeline Run Log — {ts}

## Run Metadata
- Pipeline: OCR + ingestion + eval
- Node: local edge (this machine)
- Embedding: nomic-embed-text via Forge Ollama (768-dim, CPU)
- Vector store: ChromaDB (local, ~/.mempalace/palace)
- Context FS: OpenViking viking:// (L0/L1/L2 tiers)

## Results
- Documents processed: 12
- Drawers filed: 24
- Avg extraction confidence: 0.87
- Escalations raised: 1 (low-confidence entity extraction on doc 7)
- Sync to viking://: OK

## Anomalies
- Doc 7 entity extraction confidence 0.41 below floor 0.55 — escalated.
- No cloud endpoints contacted during this run.
"""
    )
    results.append(check("sample log written", log.exists(), log.name))

    # --- 2. MemPalace ingestion -------------------------------------------
    print("\n[2/5] MemPalace ingestion (mine)")
    mine = run([str(VENV_PY), "-m", "mempalace", "mine", str(INGEST_DIR), "--wing", "pipeline-runs"])
    ok = "Drawers filed: 0" not in mine.stdout and "Done" in mine.stdout
    results.append(check("mine completed", ok, mine.stdout.strip().splitlines()[-1] if mine.stdout.strip() else mine.stderr[-200:]))

    import chromadb
    _client = chromadb.PersistentClient(path=str(Path.home() / ".mempalace" / "palace"))
    _col = _client.get_collection("mempalace_drawers")
    _data = _col.get(include=["metadatas"])
    _wings = {m.get("wing") for m in _data["metadatas"]}
    results.append(check("palace status shows new wing", "pipeline-runs" in _wings, f"wings={sorted(_wings)}"))

    # --- 3. Bridge sync ----------------------------------------------------
    print("\n[3/5] Bridge sync (MemPalace -> viking://)")
    bridge = run([str(VENV_PY), str(BRIDGE)])
    results.append(check("bridge ran", bridge.returncode == 0, bridge.stdout.strip().splitlines()[-1] if bridge.stdout.strip() else bridge.stderr[-200:]))
    results.append(check("semantic processing complete", "COMPLETE" in bridge.stdout, ""))

    # --- 4. Agent retrieval via viking:// ----------------------------------
    print("\n[4/5] Agent retrieval via viking://")
    from openviking_sdk import SyncHTTPClient

    client = SyncHTTPClient(url=VIKING_URL, api_key=USER_KEY)
    client.initialize()
    try:
        # Semantic find (agent-style retrieval)
        found = client.find(query="pipeline run extraction confidence", target_uri="viking://resources/mempalace", limit=3)
        resources = found.get("resources", []) if isinstance(found, dict) else []
        results.append(check("semantic find returned hits", len(resources) > 0, f"{len(resources)} hits"))

        # L0 abstract
        l0_ok = any((r.get("abstract") or "") for r in resources)
        results.append(check("L0 abstract present", l0_ok))

        # L2 detail read
        l2_uri = None
        for r in resources:
            uri = r.get("uri", "")
            if uri.endswith(".md") and not uri.endswith((".abstract.md", ".overview.md")):
                l2_uri = uri
                break
        if not l2_uri:
            # fall back: walk the tree for a leaf .md
            ls = client.ls(uri="viking://resources/mempalace/pipeline-runs")
            items = ls if isinstance(ls, list) else ls.get("entries") or ls.get("items") or []
            for it in items:
                u = it.get("uri") or it.get("path") or ""
                if u.endswith(".md") and not u.endswith((".abstract.md", ".overview.md")):
                    l2_uri = u
                    break
        if l2_uri:
            content = client.read(uri=l2_uri)
            results.append(check("L2 detail retrievable", len(content) > 100, f"{len(content)} chars from {l2_uri}"))
        else:
            results.append(check("L2 detail retrievable", False, "no leaf .md found"))
    finally:
        client.close()

    # --- 5. Zero cloud telemetry -------------------------------------------
    print("\n[5/5] Zero cloud telemetry check")
    ov_conf = json.loads((Path.home() / ".openviking" / "ov.conf").read_text())
    endpoints = []
    for section in ("embedding", "vlm"):
        for key in ("api_base",):
            ep = ov_conf.get(section, {}).get("dense" if section == "embedding" else "", {}).get(key, "")
            if section == "vlm":
                ep = ov_conf.get(section, {}).get(key, "")
            if ep:
                endpoints.append(ep)
    all_local = all(any(ep.replace("http://", "").replace("https://", "").startswith(n) for n in LOCAL_NETS) for ep in endpoints)
    results.append(check("all model endpoints local", all_local, ", ".join(endpoints)))

    mp_conf = json.loads((Path.home() / ".mempalace" / "config.json").read_text())
    results.append(check("MemPalace palace local", str(mp_conf["palace_path"]).startswith("/home/"), mp_conf["palace_path"]))

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 60)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())