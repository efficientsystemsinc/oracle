"""Synthetic judge-training pairs with labels correct by construction.
For each seed fact, gpt-5.5 writes: (a) a plausible NEWER version of the same
truth (label: supersedes), (b) a CONTRADICTING same-scope claim (label:
contradicts), (c) a DIFFERENT-ASPECT fact about the same system (label: none).
Output: judge_synth.jsonl in the same schema as judge_train.jsonl
(labels: REOPEN==pair where supersession would be WRONG -> from (c)+(b);
 UPHOLD==true replacement -> from (a); REOPEN_CONTRADICT -> from (b)).

usage: python3 gen_judge_synth.py [n_seeds=8000]   (reads ~/.oracle/oracle.db + ~/.oracle/azure.key; writes ~/.oracle/judge_synth.jsonl)
"""
import json, os, sqlite3, random, urllib.request, time, sys
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["ORACLE_LLM_KEY"]
URL = os.environ["ORACLE_LLM_URL"]
SYS = """For EACH numbered engineering fact, invent three related statements:
- "newer": the SAME truth at a later date after a plausible change — it genuinely REPLACES the original (config moved, number improved, status changed).
- "contra": a same-scope claim that CONTRADICTS the original (both cannot be current).
- "aspect": a TRUE-sounding fact about the same system that covers a DIFFERENT aspect (does not replace or contradict).
Keep each statement standalone, specific, same style as the input. Return JSON:
{"items":[{"idx":int,"newer":str,"contra":str,"aspect":str}]}"""

def call(batch):
    lines = "\n".join(f"{i}: {s}" for i, s in enumerate(batch))
    body = json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":lines}],
                       "response_format":{"type":"json_object"},"max_completion_tokens":16000,
                       "reasoning_effort":"low"}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=body, headers={"api-key":KEY,"Content-Type":"application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=240))
            return json.loads(r["choices"][0]["message"]["content"]).get("items", [])
        except Exception as e:
            if attempt == 2: print("batch failed:", e, file=sys.stderr); return []
            time.sleep(5 * (attempt + 1))

def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    db = sqlite3.connect(os.path.expanduser('~/.oracle/oracle.db'))
    rows = db.execute("""SELECT statement, kind, COALESCE(repo,''), valid_from,
        COALESCE(evidence,'asserted') FROM facts WHERE superseded_at IS NULL
        AND length(statement) > 60 ORDER BY RANDOM() LIMIT ?""", (n_seeds,)).fetchall()
    random.seed(7)
    out = open(os.path.expanduser('~/.oracle/judge_synth.jsonl'), 'w')
    B = 8
    batches = [rows[i:i+B] for i in range(0, len(rows), B)]
    done = 0
    def work(batch):
        stmts = [b[0] for b in batch]
        return batch, call(stmts)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for batch, items in ex.map(work, batches):
            by = {it.get("idx"): it for it in items if isinstance(it, dict)}
            for i, (stmt, kind, repo, vf, ev) in enumerate(batch):
                it = by.get(i)
                if not it: continue
                base = dict(old=stmt, old_kind=kind, old_repo=repo, old_date=vf, old_ev=ev,
                            new_kind=kind, new_repo=repo, new_date=vf + 14*86400, new_ev="asserted")
                for key, label in [("newer","UPHOLD"), ("contra","REOPEN_CONTRADICT"), ("aspect","REOPEN")]:
                    s = (it.get(key) or "").strip()
                    if len(s) > 30:
                        out.write(json.dumps(dict(base, new=s, label=label))+"\n")
            done += 1
            if done % 50 == 0: print(f"{done}/{len(batches)} batches", flush=True)
    print("SYNTH_DONE")

if __name__ == "__main__":
    main()
