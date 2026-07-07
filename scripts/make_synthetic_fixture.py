#!/usr/bin/env python3
"""Synthetic regression fixture for oracle.

Fabricates ~500 plausible but entirely FAKE facts (fake projects, fake boxes,
TEST-NET IPs) into testdata/fixture.db: live facts, supersession chains,
entities, co-mention edges, ask traces, and fact vectors — so the CI selfeval
gate runs on public data with zero private session content.

Vectors are a deterministic bag-of-tokens hash embedding (512-dim, unit norm).
The SAME function serves query embeddings via `--serve PORT` (an
OpenAI-compatible /embeddings stub), so cosine search in CI is meaningful:
identical text -> identical vector; shared tokens -> high similarity.

Usage:
  python3 scripts/make_synthetic_fixture.py --oracle ./oracle-bin --out testdata/fixture.db
  python3 scripts/make_synthetic_fixture.py --serve 8901   # stub embed server

Regenerate thresholds after: run selfeval on the fixture with the stub server
up (see .github/workflows/ci.yml) and set testdata/thresholds.json a few
points below the measured rates.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time

DIMS = 512
DAY = 86400.0
BASE_TS = time.time() - 200 * DAY  # corpus spans the ~200 days before generation


# ---------------------------------------------------------------- embedding

def _token_vec(tok: str):
    rng = random.Random(hashlib.sha256(tok.encode()).digest())
    return [rng.gauss(0, 1) for _ in range(DIMS)]


def hash_embed(text: str):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    v = [0.0] * DIMS
    for t in toks:
        tv = _token_vec(t)
        for i in range(DIMS):
            v[i] += tv[i]
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def vec_blob(v):
    return struct.pack("<%df" % len(v), *v)


# ---------------------------------------------------------------- stub server

def serve(port: int):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inputs = body["input"]
            if isinstance(inputs, str):
                inputs = [inputs]
            data = [{"index": i, "embedding": hash_embed(s)} for i, s in enumerate(inputs)]
            out = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    print(f"stub embed server on :{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()


# ---------------------------------------------------------------- fake corpus

PROJECTS = {
    "quasar": "API gateway service",
    "meadow": "CI runner fleet",
    "lighthouse": "metrics dashboard",
    "driftwood": "object storage proxy",
    "tumbleweed": "batch ETL pipeline",
    "foxglove": "auth service",
    "peregrine": "search indexer",
    "bramble": "feature-flag daemon",
}
BOXES = ["atlas-01", "nimbus-02", "cinder-03", "harbor-04", "mesa-05"]
IPS = ["192.0.2.10", "192.0.2.21", "198.51.100.7", "198.51.100.42", "203.0.113.5"]
PEOPLE = ["alex", "jordan", "casey", "riley"]
DBS = ["postgres 16", "redis 7", "sqlite", "clickhouse"]


def corpus(rng):
    """Returns (live_facts, chains, probes).
    Each fact: dict(statement, kind, repo, entities, files).
    Each probe: (question, expected-regex) — hit@5 rows for eval/probes.tsv."""
    live, chains, probes = [], [], []
    mk = lambda st, kind, repo, ents, files=(): dict(
        statement=st, kind=kind, repo=repo, entities=list(ents), files=list(files))

    for proj, desc in PROJECTS.items():
        box = rng.choice(BOXES)
        ip = rng.choice(IPS)
        port_old = rng.randrange(7000, 8000)
        port_new = port_old + 1000
        db = rng.choice(DBS)
        person = rng.choice(PEOPLE)

        # supersession chains (port move, box move, version bump)
        chains.append((
            mk(f"{proj} ({desc}) serves on {box} port {port_old}", "fact", proj, [proj, box]),
            mk(f"{proj} ({desc}) serves on {box} port {port_new} after the ingress rework", "fact", proj, [proj, box]),
        ))
        chains.append((
            mk(f"{proj} deploys go via meadow runner pool a with 4 workers", "fact", proj, [proj, "meadow"]),
            mk(f"{proj} deploys via meadow runner pool b with 8 workers since the capacity bump", "fact", proj, [proj, "meadow"]),
        ))
        day = rng.choice(["monday", "tuesday", "wednesday", "thursday", "friday"])
        chains.append((
            mk(f"the {proj} backing store for {proj} is {db} on {ip}", "fact", proj, [proj, box]),
            mk(f"the {proj} backing store for {proj} is now {rng.choice(DBS)} on {ip} after the {proj} storage migration", "fact", proj, [proj, box]),
        ))
        chains.append((
            mk(f"{proj} release cadence: {proj} ships weekly, cut by {person} on {day}s", "fact", proj, [proj, person]),
            mk(f"{proj} release cadence: {proj} now ships daily, cut automatically by meadow — {person} no longer cuts {proj} releases", "fact", proj, [proj, "meadow", person]),
        ))
        chains.append((
            mk(f"todo: {proj} tls certs on {box} expire and must be rotated by hand", "todo", proj, [proj, box]),
            mk(f"{proj} tls certs on {box} auto-rotate via the cert daemon; the manual {proj} rotation todo is done", "status", proj, [proj, box]),
        ))

        # live-only facts
        live += [
            mk(f"decision: {proj} keeps a single-writer queue instead of sharding — simpler recovery won the review", "decision", proj, [proj]),
            mk(f"gotcha: {proj} healthcheck lies during warmup; wait for /ready not /health", "gotcha", proj, [proj], [f"{proj}/server/health.go"]),
            mk(f"{proj} p99 latency benchmark is {rng.randrange(40, 400)}ms at {rng.randrange(2, 20)}k rps", "fact", proj, [proj]),
            mk(f"{person} prefers {proj} changes shipped behind bramble flags first", "preference", proj, [proj, "bramble", person]),
            mk(f"todo: {proj} still needs retry jitter on the {rng.choice(BOXES)} client path", "todo", proj, [proj]),
            mk(f"{proj} logs rotate at 512mb on {box}; older logs gzip to /var/log/{proj}/archive", "fact", proj, [proj, box]),
            mk(f"{proj} staging runs one replica on {rng.choice(BOXES)} — do not benchmark against staging", "gotcha", proj, [proj]),
            mk(f"decision: {proj} pins its client library to v{rng.randrange(2, 9)} until the breaking pagination change is absorbed", "decision", proj, [proj]),
            mk(f"{proj} alert threshold is {rng.randrange(85, 99)}% disk on {rng.choice(BOXES)}, paging {person}", "fact", proj, [proj, person]),
            mk(f"gotcha: {proj} integration tests need the {rng.choice(DBS)} container warmed first or the first run flakes", "gotcha", proj, [proj], [f"{proj}/tests/setup.sh"]),
            mk(f"{proj} config lives in /etc/{proj}/config.toml and hot-reloads on sighup", "fact", proj, [proj], [f"{proj}/cmd/reload.go"]),
            mk(f"decision: {proj} rejected kafka for its queue — {db} outbox polling is enough at current volume", "decision", proj, [proj]),
            mk(f"{proj} nightly backup lands on {rng.choice(BOXES)} at 03:15 utc, retained {rng.randrange(7, 30)} days", "fact", proj, [proj]),
            mk(f"{person} owns the {proj} oncall rotation this quarter", "status", proj, [proj, person]),
            mk(f"gotcha: {proj} silently drops requests over {rng.randrange(1, 16)}mb — raise the body limit before bulk imports", "gotcha", proj, [proj]),
            mk(f"todo: {proj} needs a runbook for the {rng.choice(BOXES)} failover path", "todo", proj, [proj]),
            mk(f"{proj} build takes {rng.randrange(2, 14)} minutes on meadow; cache hits cut it to under one", "fact", proj, [proj, "meadow"]),
            mk(f"decision: {proj} logs structured json only — the plaintext logger was removed", "decision", proj, [proj]),
            mk(f"{proj} rate limit is {rng.randrange(100, 900)} rps per token, enforced at foxglove", "fact", proj, [proj, "foxglove"]),
        ]

        if len(probes) < 16:
            probes += [
                (f"which port does {proj} serve on now", f"port {port_new}"),
                (f"what is the {proj} backing store today", f"{proj} backing store.*storage migration"),
            ]

    # cross-project facts
    for i in range(150):
        a, b = rng.sample(list(PROJECTS), 2)
        live.append(mk(
            f"{a} calls {b} over grpc with a {rng.randrange(1, 9)}s deadline (case {i})",
            "fact", a, [a, b]))
    return live, chains, probes


# ---------------------------------------------------------------- build

def build(oracle_bin: str, out: str):
    rng = random.Random(20260708)
    tmp = tempfile.mkdtemp(prefix="oracle-synth-")
    env = dict(os.environ, ORACLE_HOME=tmp)
    subprocess.run([oracle_bin, "init"], check=True, env=env, stdout=subprocess.DEVNULL)
    dbp = os.path.join(tmp, "oracle.db")
    db = sqlite3.connect(dbp)

    live, chains, probes = corpus(rng)
    ent_ids = {}

    def entity_id(name):
        if name not in ent_ids:
            cur = db.execute("INSERT INTO entities(name, display, seen_count, last_seen) VALUES(?,?,?,?)",
                             (name, name, 1, BASE_TS))
            ent_ids[name] = cur.lastrowid
        return ent_ids[name]

    def insert_fact(f, recorded, valid, superseded_at=None, superseded_by=None):
        cur = db.execute(
            """INSERT INTO facts(statement, kind, repo, entities, files, confidence, mass,
               recorded_at, valid_from, superseded_at, superseded_by, src_path, src_session)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f["statement"], f["kind"], f["repo"], json.dumps(f["entities"]),
             json.dumps(f["files"]), round(rng.uniform(0.6, 0.95), 2), 1.0,
             recorded, valid, superseded_at, superseded_by,
             f"synthetic/{f['repo']}.jsonl", "synthetic-session"))
        fid = cur.lastrowid
        for e in f["entities"]:
            db.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES(?,?)",
                       (fid, entity_id(e)))
        for i, a in enumerate(f["entities"]):
            for b in f["entities"][i + 1:]:
                x, y = sorted((entity_id(a), entity_id(b)))
                db.execute("""INSERT INTO entity_edges(a,b,count,last_seen) VALUES(?,?,1,?)
                              ON CONFLICT(a,b) DO UPDATE SET count=count+1""", (x, y, recorded))
        db.execute("INSERT INTO fact_vecs(fact_id, vec) VALUES(?,?)",
                   (fid, vec_blob(hash_embed(f["statement"]))))
        return fid

    ts = BASE_TS
    fact_ids = []
    for old, new in chains:
        t_old = ts
        t_new = ts + rng.uniform(20, 90) * DAY
        new_id_holder = insert_fact(new, t_new, t_new)
        old_id = insert_fact(old, t_old, t_old, superseded_at=t_new, superseded_by=new_id_holder)
        db.execute("INSERT OR IGNORE INTO edges(src,dst,type,recorded_at) VALUES(?,?,?,?)",
                   (new_id_holder, old_id, "supersedes", t_new))
        fact_ids.append(new_id_holder)
        ts += rng.uniform(0.5, 2.0) * DAY

    for f in live:
        t = BASE_TS + rng.uniform(0, 150) * DAY
        fact_ids.append(insert_fact(f, t, t))

    # ask traces with citations (feeds selfeval citation replay)
    cited_pool = rng.sample(fact_ids, 30)
    for i, fid in enumerate(cited_pool):
        stmt = db.execute("SELECT statement FROM facts WHERE id=?", (fid,)).fetchone()[0]
        q = "what do we know about " + " ".join(stmt.split()[:6])
        db.execute("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)",
                   (BASE_TS + 160 * DAY + i, "ask", q, json.dumps({"cited": [fid]})))

    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    # strip WAL sidecars: reopen in DELETE journal mode
    db = sqlite3.connect(dbp)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("VACUUM")
    db.close()
    shutil.copyfile(dbp, out)
    shutil.rmtree(tmp)

    probes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), "eval", "probes.tsv")
    with open(probes_path, "w") as f:
        for q, expect in probes:
            f.write(f"{q}\t{expect}\n")

    n = sqlite3.connect(out).execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(f"synthetic fixture: {out} ({n} facts, {len(chains)} chains, 30 ask traces, {len(probes)} probes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="./oracle-bin", help="path to the oracle binary (for schema init)")
    ap.add_argument("--out", default="testdata/fixture.db")
    ap.add_argument("--serve", type=int, help="run as a stub OpenAI-compatible embeddings server on this port")
    a = ap.parse_args()
    if a.serve:
        serve(a.serve)
    else:
        build(a.oracle, a.out)


if __name__ == "__main__":
    sys.exit(main())
