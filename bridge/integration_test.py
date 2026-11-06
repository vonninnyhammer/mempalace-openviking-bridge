#!/usr/bin/env python3
"""
integration_test.py — end-to-end validation of the dual-layer architecture.

Tests the contract that is actually deployed:

  Layer 1 (MemPalace)  full-corpus memory. Every drawer, any importance.
  Layer 2 (OpenViking) curated execution-time context. The bridge pushes a
                      capped, importance-filtered slice (--max-drawers,
                      --min-importance) because the L0/L1 VLM pass costs
                      ~12s per resource on CPU. viking:// is therefore NOT
                      expected to contain freshly mined drawers, and this
                      test asserts that boundary rather than fighting it.

Two things an earlier version of this test got wrong, both of which are now
regression checks:

  1. It mined the real `ingest/` tree, which recursively swept up the live
     `ingest/opencode/` capture dropbox and the exporter's own bookkeeping
     JSON (.export-state.meta.json, sessions-index.meta.json) into a junk
     wing. The dropbox is now an isolated temp dir and the test wing is
     pruned in a finally block, so a run leaves no residue.

  2. It asserted the test wing appeared under viking://, which cannot happen
     under the curation caps. It also crashed with an unhandled NotFoundError
     instead of reporting a failed check.

Exit code 0 = PASS, 1 = FAIL.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

VENV_PY = Path.home() / ".mempalace-venv" / "bin" / "python"
VENV_BIN = Path.home() / ".mempalace-venv" / "bin"
BRIDGE = Path.home() / "mempalace-viking" / "bridge" / "mempalace_viking_bridge.py"
STAGING = Path.home() / "mempalace-viking" / "bridge" / "staging" / "mempalace"
VIKING_ROOT = "viking://resources/mempalace"
VIKING_URL = "http://localhost:1933"
USER_KEY = (Path.home() / ".openviking" / ".user_key").read_text().strip()

# Must match the bridge's production defaults, or the curation checks are vacuous.
BRIDGE_MAX_DRAWERS = 60
BRIDGE_MIN_IMPORTANCE = 2.0

# Local-only network ranges (zero cloud telemetry).
LOCAL_NETS = ("127.", "10.", "192.168.", "100.86.", "fd7a:", "fe80:", "::1")


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_mine(src: Path, wing: str, attempts: int = 4) -> subprocess.CompletedProcess:
    """Mine, retrying briefly if the MCP server happens to hold the palace lock.

    The lock is normally free — the 10-minute capture cycle mines the same
    palace while this server is up — but a manual mine can collide with a
    reconnect. A collision is transient, not a failure.
    """
    for i in range(attempts):
        p = run([str(VENV_BIN / "mempalace"), "mine", str(src), "--wing", wing])
        if "held by PID" not in (p.stdout + p.stderr):
            return p
        if i < attempts - 1:
            time.sleep(5)
    return p


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def skip(name: str, reason: str) -> None:
    print(f"  [SKIP] {name} — {reason}")


def mcp_lock_holder() -> str | None:
    """Return the PID of an MCP server holding the palace lock, else None.

    `mempalace mine` and `mempalace sync` take an exclusive palace lock, and
    the mempalace MCP server holds it for its whole lifetime. So any test run
    from inside an agent session that has the MCP server connected cannot
    write to the palace. Read paths (chroma directly, OpenViking) are
    unaffected. The honest outcome there is SKIP, not FAIL.
    """
    probe = Path(tempfile.mkdtemp(prefix="mempalace-lockprobe-"))
    try:
        (probe / "probe.txt").write_text("lock probe\n")
        p = run([str(VENV_BIN / "mempalace"), "mine", str(probe), "--wing", "lock-probe"], timeout=120)
        blob = p.stdout + p.stderr
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    if "held by PID" not in blob:
        return None
    pid = blob.split("held by PID", 1)[1].split()[0]
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return None
    return pid if "mempalace-mcp" in cmd else None


def http_ls(uri: str) -> list[dict]:
    """Raw HTTP ls. The SDK strips `abstract`; the API returns it."""
    req = urllib.request.Request(
        f"{VIKING_URL}/api/v1/fs/ls?uri={urllib.parse.quote(uri)}",
        headers={"X-API-Key": USER_KEY},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("result", [])


def staged_sample_query() -> str:
    """Build a retrieval query from content actually staged for viking://.

    Deriving the query from the live staging tree means the retrieval check
    cannot rot as the palace's top-importance drawers change over time.
    """
    for f in sorted(STAGING.rglob("*.md")):
        body = "\n".join(
            ln for ln in f.read_text(errors="replace").splitlines()
            if not ln.startswith("<!--")
        ).strip()
        words = [w for w in body.split() if w.isalpha() and len(w) > 3]
        if len(words) >= 8:
            return " ".join(words[:12])
    return "pipeline run extraction confidence"


def cleanup(dropbox: Path, wing: str) -> list[str]:
    """Remove the temp dropbox and prune every drawer mined from it.

    Deleting the source files first is what makes the prune effective:
    `mempalace sync` drops drawers whose source_file no longer resolves.
    """
    notes = []
    shutil.rmtree(dropbox, ignore_errors=True)
    prune = run([str(VENV_BIN / "mempalace"), "sync", "--wing", wing, "--apply"], timeout=300)
    if prune.returncode != 0:
        notes.append(f"prune exited {prune.returncode}: {prune.stderr.strip()[-200:]}")
    return notes


def drawer_count(wing: str) -> int:
    import chromadb

    col = chromadb.PersistentClient(
        path=str(Path.home() / ".mempalace" / "palace")
    ).get_collection("mempalace_drawers")
    return len(col.get(include=["metadatas"], where={"wing": wing})["metadatas"])


def main() -> int:
    results: list[bool | None] = []
    skipped = 0
    dropbox = Path(tempfile.mkdtemp(prefix="mempalace-itest-"))
    wing = "itest-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    holder = mcp_lock_holder()
    lock_note = (
        f"palace lock held by mempalace-mcp PID {holder} — "
        "run from a shell without the MCP server connected"
        if holder
        else ""
    )

    print("=" * 60)
    print("DUAL-LAYER INTEGRATION TEST")
    print(f"isolated dropbox: {dropbox}")
    print(f"throwaway wing:   {wing}")
    if holder:
        print(f"LOCK:             {lock_note}")
    print("=" * 60)

    try:
        # --- 1. Sample log in the isolated dropbox -------------------------
        print("\n[1/6] Sample log")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = dropbox / f"pipeline-run-{ts}.txt"
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

        # --- 2. MemPalace ingestion ----------------------------------------
        print("\n[2/6] MemPalace ingestion (mine)")
        if holder:
            skip("mine filed drawers into throwaway wing", lock_note)
            skipped += 1
        else:
            mine = run_mine(dropbox, wing)
            filed = drawer_count(wing)
            results.append(
                check(
                    "mine filed drawers into throwaway wing",
                    filed > 0,
                    f"{filed} drawers" if filed else mine.stdout.strip()[-200:],
                )
            )

        # --- 3. Curation contract ------------------------------------------
        # The palace is full-corpus; viking:// is capped. Assert both halves.
        print("\n[3/6] Curation contract")
        staged = sorted(STAGING.rglob("*.md"))
        results.append(
            check(
                "staging respects max-drawers cap",
                0 < len(staged) <= BRIDGE_MAX_DRAWERS,
                f"{len(staged)} staged, cap {BRIDGE_MAX_DRAWERS}",
            )
        )
        results.append(
            check(
                "unranked wing stays out of viking:// (min-importance floor)",
                not any(wing in str(f) for f in staged),
                f"floor {BRIDGE_MIN_IMPORTANCE}, fresh drawers score 0.0",
            )
        )

        # --- 4. Bridge sync --------------------------------------------------
        print("\n[4/6] Bridge sync (MemPalace -> viking://)")
        bridge = run([str(VENV_PY), str(BRIDGE)])
        results.append(
            check(
                "bridge exited 0",
                bridge.returncode == 0,
                bridge.stdout.strip().splitlines()[-1] if bridge.stdout.strip() else bridge.stderr[-200:],
            )
        )
        if holder and "no changes since last sync" in bridge.stdout:
            # Expected: nothing new was mined, so the bridge correctly
            # early-returns without a sync submit or a semantic pass.
            skip("semantic processing complete", "no staged changes to push")
            skipped += 1
        else:
            results.append(
                check(
                    "semantic processing complete",
                    "COMPLETE" in bridge.stdout,
                    "TIMEOUT" if "TIMEOUT" in bridge.stdout else "",
                )
            )

        # --- 5. Agent retrieval via viking:// -------------------------------
        print("\n[5/6] Agent retrieval via viking://")
        from openviking_sdk import SyncHTTPClient

        query = staged_sample_query()
        client = SyncHTTPClient(url=VIKING_URL, api_key=USER_KEY)
        client.initialize()
        try:
            found = client.find(query=query, target_uri=VIKING_ROOT, limit=3)
            resources = found.get("resources", []) if isinstance(found, dict) else []
            results.append(
                check("semantic find returned hits", len(resources) > 0, f"{len(resources)} hits for {query!r}")
            )

            l0_uri = None
            for r in resources:
                if (r.get("abstract") or "").strip():
                    l0_uri = r.get("uri", "")
                    break
            if l0_uri is None:
                for item in http_ls(VIKING_ROOT):
                    if (item.get("abstract") or "").strip():
                        l0_uri = item.get("uri", "")
                        break
            results.append(check("L0 abstract present", l0_uri is not None, l0_uri or "none"))

            # L2 detail. Walk the real tree instead of assuming a test wing
            # exists, and treat a missing node as a failed check, not a crash.
            l2_uri = next(
                (
                    r.get("uri", "")
                    for r in resources
                    if r.get("uri", "").endswith(".md")
                    and not r["uri"].endswith((".abstract.md", ".overview.md"))
                ),
                None,
            )
            if l2_uri is None:
                for item in http_ls(VIKING_ROOT):
                    if not item.get("isDir"):
                        u = item.get("uri", "")
                        if u.endswith(".md") and not u.endswith((".abstract.md", ".overview.md")):
                            l2_uri = u
                            break
            if l2_uri:
                content = client.read(uri=l2_uri)
                results.append(
                    check("L2 detail retrievable", len(content) > 100, f"{len(content)} chars from {l2_uri}")
                )
            else:
                results.append(check("L2 detail retrievable", False, "no leaf .md under viking://"))
        finally:
            client.close()

        # --- 6. Zero cloud telemetry ----------------------------------------
        print("\n[6/6] Zero cloud telemetry check")
        ov_conf = json.loads((Path.home() / ".openviking" / "ov.conf").read_text())
        endpoints = []
        for section in ("embedding", "vlm"):
            key = "dense" if section == "embedding" else ""
            ep = ov_conf.get(section, {}).get(key, {}).get("api_base", "") if key else ov_conf.get(section, {}).get("api_base", "")
            if ep:
                endpoints.append(ep)
        all_local = all(
            any(ep.replace("http://", "").replace("https://", "").startswith(n) for n in LOCAL_NETS)
            for ep in endpoints
        )
        results.append(check("all model endpoints local", all_local, ", ".join(endpoints)))

        mp_conf = json.loads((Path.home() / ".mempalace" / "config.json").read_text())
        results.append(
            check("MemPalace palace local", str(mp_conf["palace_path"]).startswith("/home/"), mp_conf["palace_path"])
        )

    finally:
        # --- 7. Cleanup -----------------------------------------------------
        print("\n[cleanup] removing test artifacts")
        for note in cleanup(dropbox, wing):
            print(f"  [WARN] {note}")
        left = drawer_count(wing)
        if holder and left == 0:
            # Nothing was mined, so there is nothing to prune.
            skip("pruned test wing", "mine was skipped, nothing to prune")
            skipped += 1
        else:
            results.append(check("no residue in palace", left == 0, f"{left} drawers left in {wing}"))

    graded = [r for r in results if r is not None]
    passed = sum(1 for r in graded if r)
    total = len(graded)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} checks passed" + (f", {skipped} skipped" if skipped else ""))
    if holder:
        print(f"NOTE: palace lock was held by mempalace-mcp PID {holder}.")
        print("      Write-path checks were skipped, not passed. To exercise them,")
        print("      run this from a shell with no mempalace MCP server connected.")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
