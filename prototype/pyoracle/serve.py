"""HTTP surface + ingest loop. `oracle up` runs both."""
import json
import threading
import time

from fastapi import FastAPI, HTTPException

from . import db, graph

app = FastAPI(title="oracle", version="0.1.0")
INGEST_INTERVAL = 300
_loop_state = {"running": False, "last": None}


@app.get("/health")
def health():
    con = db.connect()
    try:
        n = con.execute("SELECT COUNT(*) c FROM facts WHERE superseded_at IS NULL").fetchone()["c"]
        total = con.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
        last = con.execute("SELECT v FROM meta WHERE k='last_cycle'").fetchone()
        return {"ok": True, "live_facts": n, "total_facts": total,
                "last_cycle": json.loads(last["v"]) if last else None,
                "ingest_loop": _loop_state["running"]}
    finally:
        con.close()


@app.get("/query")
def query(q: str, repo: str | None = None, k: int = 10, as_of: float | None = None):
    con = db.connect()
    try:
        return {"q": q, "hits": graph.search(con, q, repo=repo, k=k, as_of=as_of)}
    finally:
        con.close()


@app.get("/brief")
def brief_ep(repo: str | None = None, k: int = 30):
    con = db.connect()
    try:
        return {"repo": repo, "brief": graph.brief(con, repo, k)}
    finally:
        con.close()


@app.get("/facts/{fact_id}")
def fact(fact_id: int):
    con = db.connect()
    try:
        r = con.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        if not r:
            raise HTTPException(404)
        d = dict(r)
        d["superseded_by_chain"] = []
        cur = r
        while cur and cur["superseded_by"]:
            cur = con.execute("SELECT * FROM facts WHERE id=?", (cur["superseded_by"],)).fetchone()
            if cur:
                d["superseded_by_chain"].append({"id": cur["id"], "statement": cur["statement"]})
        return d
    finally:
        con.close()


@app.post("/cycle")
def run_cycle(max_calls: int = 20, since_days: float = 7):
    return graph.cycle(max_calls=max_calls, since_days=since_days)


def ingest_forever(max_calls: int = 20, since_days: float = 7):
    _loop_state["running"] = True
    while True:
        try:
            _loop_state["last"] = graph.cycle(max_calls=max_calls, since_days=since_days)
        except Exception as e:
            _loop_state["last"] = {"fatal": repr(e)}
        time.sleep(INGEST_INTERVAL)


def up(port: int = 4141, max_calls: int = 20):
    t = threading.Thread(target=ingest_forever, args=(max_calls,), daemon=True)
    t.start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
