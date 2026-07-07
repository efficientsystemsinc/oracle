"""Graph ops: ingest cycle (watch -> extract -> upsert w/ supersede),
decay-weighted search, briefs, as-of queries."""
import datetime
import json
import math
import os
import re
import time

from . import db, extract, watch

# half-life (days) per kind — status/todo rot fast, preferences barely
HALF_LIFE = {"status": 7, "todo": 14, "fact": 60, "decision": 120, "gotcha": 120, "preference": 365}
MASS_EPS = 0.05


def mass_now(row, now: float) -> float:
    hl = HALF_LIFE.get(row["kind"], 60) * 86400
    age = max(0.0, now - row["valid_from"])
    return MASS_EPS + row["mass"] * math.exp(-0.693 * age / hl)


def _fts_query(text: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_.-]{2,}", text)[:12]
    return " OR ".join(f'"{t}"' for t in toks) if toks else '""'


def search(con, q: str, repo: str | None = None, k: int = 10,
           as_of: float | None = None, reinforce: bool = True) -> list[dict]:
    now = time.time()
    where = "WHERE facts_fts MATCH ?"
    args: list = [_fts_query(q)]
    if as_of:
        where += " AND f.recorded_at <= ? AND (f.superseded_at IS NULL OR f.superseded_at > ?)"
        args += [as_of, as_of]
    else:
        where += " AND f.superseded_at IS NULL"
    rows = con.execute(
        f"""SELECT f.*, bm25(facts_fts, 2.0, 1.0) AS rank FROM facts_fts
            JOIN facts f ON f.id = facts_fts.rowid {where}
            ORDER BY rank LIMIT 60""", args).fetchall()
    scored = []
    for r in rows:
        s = -r["rank"] + 0.8 * math.log(mass_now(r, as_of or now)) + 0.5 * r["confidence"]
        if repo and r["repo"] == repo:
            s += 1.5
        scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    top = scored[:k]
    if reinforce and not as_of and top:
        ids = [r["id"] for _, r in top[:3]]
        con.execute(
            f"UPDATE facts SET mass = mass + 0.15, use_count = use_count + 1, last_used_at = ? "
            f"WHERE id IN ({','.join('?' * len(ids))})", [now, *ids])
    con.execute("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)",
                (now, "query", q, json.dumps([r["id"] for _, r in top])))
    con.commit()
    return [_fact_dict(r, s, now) for s, r in top]


def brief(con, repo: str | None, k: int = 30) -> dict:
    now = time.time()
    where, args = "WHERE superseded_at IS NULL", []
    if repo:
        where += " AND repo = ?"
        args.append(repo)
    rows = con.execute(f"SELECT * FROM facts {where}", args).fetchall()
    scored = sorted(rows, key=lambda r: -(mass_now(r, now) * (0.5 + r["confidence"])))[:k]
    out: dict[str, list] = {}
    for r in scored:
        out.setdefault(r["kind"], []).append(_fact_dict(r, mass_now(r, now), now))
    return out


def _fact_dict(r, score: float, now: float) -> dict:
    return {
        "id": r["id"], "statement": r["statement"], "kind": r["kind"],
        "repo": r["repo"], "entities": json.loads(r["entities"]),
        "files": json.loads(r["files"]), "confidence": r["confidence"],
        "score": round(score, 3), "mass": round(mass_now(r, now), 3),
        "age_days": round((now - r["valid_from"]) / 86400, 1),
        "src": r["src_session"],
    }


def _link_entities(con, fact_id: int, names: list[str], now: float) -> None:
    ids = []
    for n in names:
        canon = n.strip().lower()
        if not canon:
            continue
        con.execute(
            "INSERT INTO entities(name, display, seen_count, last_seen) VALUES(?,?,1,?) "
            "ON CONFLICT(name) DO UPDATE SET seen_count = seen_count + 1, last_seen = ?",
            (canon, n.strip(), now, now))
        eid = con.execute("SELECT id FROM entities WHERE name = ?", (canon,)).fetchone()["id"]
        con.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES(?,?)",
                    (fact_id, eid))
        ids.append(eid)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            lo, hi = min(a, b), max(a, b)
            con.execute(
                "INSERT INTO entity_edges(a, b, count, last_seen) VALUES(?,?,1,?) "
                "ON CONFLICT(a, b) DO UPDATE SET count = count + 1, last_seen = ?",
                (lo, hi, now, now))


def entity(con, name: str, k: int = 20) -> dict:
    canon = name.strip().lower()
    e = con.execute("SELECT * FROM entities WHERE name = ?", (canon,)).fetchone()
    if not e:
        # prefix match fallback is explicit, not silent: report what matched
        e = con.execute("SELECT * FROM entities WHERE name LIKE ? ORDER BY seen_count DESC",
                        (canon + "%",)).fetchone()
    if not e:
        return {"entity": None}
    now = time.time()
    rows = con.execute(
        """SELECT f.* FROM fact_entities fe JOIN facts f ON f.id = fe.fact_id
           WHERE fe.entity_id = ? AND f.superseded_at IS NULL""", (e["id"],)).fetchall()
    facts = sorted((_fact_dict(r, mass_now(r, now), now) for r in rows),
                   key=lambda d: -d["mass"])[:k]
    co = con.execute(
        """SELECT en.display, ee.count FROM entity_edges ee
           JOIN entities en ON en.id = CASE WHEN ee.a = ? THEN ee.b ELSE ee.a END
           WHERE ee.a = ? OR ee.b = ? ORDER BY ee.count DESC LIMIT 15""",
        (e["id"], e["id"], e["id"])).fetchall()
    return {"entity": e["display"], "seen_count": e["seen_count"], "facts": facts,
            "co_mentioned": [{"name": r["display"], "count": r["count"]} for r in co]}


def _supersede_candidates(con, fact: dict, repo: str) -> list[dict]:
    rows = con.execute(
        """SELECT f.id, f.statement FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid
           WHERE facts_fts MATCH ? AND f.superseded_at IS NULL AND f.repo = ? AND f.kind = ?
           ORDER BY bm25(facts_fts) LIMIT 3""",
        (_fts_query(fact["statement"]), repo, fact["kind"])).fetchall()
    return [dict(r) for r in rows]


def ingest_chunk(con, chunk: watch.Chunk) -> int:
    """Extract + upsert one chunk. Offset advance commits atomically with facts."""
    n_new = 0
    if chunk.text.strip():
        date = datetime.datetime.fromtimestamp(chunk.event_time).strftime("%Y-%m-%d")
        facts = extract.extract_facts(chunk.text, chunk.repo, date)
        cands = {}
        for i, f in enumerate(facts):
            c = _supersede_candidates(con, f, chunk.repo)
            if c:
                cands[i] = c
        verdicts = extract.judge_supersede(facts, cands) if cands else {}
        now = time.time()
        for i, f in enumerate(facts):
            cur = con.execute(
                "INSERT INTO facts(statement, kind, repo, entities, files, confidence, "
                "recorded_at, valid_from, src_path, src_session) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (f["statement"], f["kind"], chunk.repo, json.dumps(f["entities"]),
                 json.dumps(f["files"]), f["confidence"], now, chunk.event_time,
                 chunk.path, chunk.session))
            new_id = cur.lastrowid
            _link_entities(con, new_id, f["entities"], now)
            for old_id in verdicts.get(i, []):
                con.execute(
                    "UPDATE facts SET superseded_at = ?, superseded_by = ? "
                    "WHERE id = ? AND superseded_at IS NULL", (now, new_id, old_id))
                con.execute("INSERT OR IGNORE INTO edges(src, dst, type, recorded_at) "
                            "VALUES(?,?,?,?)", (new_id, old_id, "supersedes", now))
            n_new += 1
    con.execute(
        "INSERT INTO files(path, source, repo, offset, mtime, last_scan) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET offset = excluded.offset, repo = excluded.repo, "
        "mtime = excluded.mtime, last_scan = excluded.last_scan",
        (chunk.path, chunk.source, chunk.repo, chunk.end_offset,
         chunk.event_time, time.time()))
    con.commit()
    return n_new


def cycle(max_calls: int = 20, since_days: float | None = 7) -> dict:
    """One ingest pass. max_calls caps extraction LLM calls (cost knob)."""
    con = db.connect()
    stats = {"files_seen": 0, "chunks": 0, "facts": 0, "errors": 0, "skipped_budget": False}
    calls = 0
    try:
        known = {r["path"]: r for r in con.execute("SELECT * FROM files")}
        for path, source in watch.discover(since_days):
            k = known.get(path)
            offset = k["offset"] if k else 0
            try:
                if os.path.getsize(path) <= offset:
                    continue
            except OSError:
                continue
            stats["files_seen"] += 1
            repo = k["repo"] if k and k["repo"] != "unknown" else None
            for chunk in watch.read_new(path, source, offset, repo):
                if chunk.text.strip():
                    if calls >= max_calls:
                        stats["skipped_budget"] = True
                        break
                    calls += 1
                try:
                    stats["facts"] += ingest_chunk(con, chunk)
                    stats["chunks"] += 1
                except Exception as e:  # one bad chunk must not kill the cycle
                    con.rollback()
                    stats["errors"] += 1
                    con.execute("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)",
                                (time.time(), "error", chunk.path, repr(e)[:500]))
                    con.commit()
                    break  # do not advance past a failed chunk in this file
            if stats["skipped_budget"]:
                break
        con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('last_cycle', ?)",
                    (json.dumps({"ts": time.time(), **stats}),))
        con.commit()
    finally:
        con.close()
    return stats
