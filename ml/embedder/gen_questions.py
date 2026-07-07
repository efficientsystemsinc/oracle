"""Question->fact pairs for embedder round 3: for each live fact, gpt-5.5
writes ONE natural question an engineer would ask that this fact answers.
Output: q_pairs.jsonl {"statement":..., "paraphrase": <question>} (same schema
as embed_pairs so embedder_train.py consumes it unchanged).

usage: python3 gen_questions.py   (reads ~/.oracle/oracle.db + ~/.oracle/azure.key; writes ~/.oracle/q_pairs.jsonl)"""
import json, os, sqlite3, urllib.request, time, sys
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["ORACLE_LLM_KEY"]
URL = os.environ["ORACLE_LLM_URL"]
SYS = """For EACH numbered engineering fact, write ONE natural question a developer would ask that the fact directly answers. Vary style: some terse ("meadow ssh user?"), some full sentences, some "how/why/where/what" forms. Do NOT reuse the fact's exact phrasing — ask the way someone would who doesn't know the answer. Return JSON {"items":[{"idx":int,"q":str}]}"""
def call(batch):
    lines = "\n".join(f"{i}: {s}" for i, s in enumerate(batch))
    body = json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":lines}],
                       "response_format":{"type":"json_object"},"max_completion_tokens":12000,
                       "reasoning_effort":"low"}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=body, headers={"api-key":KEY,"Content-Type":"application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=240))
            return json.loads(r["choices"][0]["message"]["content"]).get("items", [])
        except Exception as e:
            if attempt == 2: print("batch failed:", e, file=sys.stderr); return []
            time.sleep(5*(attempt+1))
def main():
    db = sqlite3.connect(os.path.expanduser('~/.oracle/oracle.db'))
    rows = [r[0] for r in db.execute("SELECT statement FROM facts WHERE superseded_at IS NULL")]
    out = open(os.path.expanduser('~/.oracle/q_pairs.jsonl'), 'w')
    B = 12
    batches = [rows[i:i+B] for i in range(0, len(rows), B)]
    done = 0
    def work(b): return b, call(b)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for batch, items in ex.map(work, batches):
            by = {it.get("idx"): it for it in items if isinstance(it, dict)}
            for i, stmt in enumerate(batch):
                it = by.get(i)
                if it and len((it.get("q") or "").strip()) > 8:
                    out.write(json.dumps({"statement": stmt, "paraphrase": it["q"].strip()})+"\n")
            done += 1
            if done % 100 == 0: print(f"{done}/{len(batches)}", flush=True)
    print("QGEN_DONE")
main()
