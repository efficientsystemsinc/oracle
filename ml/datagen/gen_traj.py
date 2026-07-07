# Generate retrieval-trajectory training data: for entity-linked fact pairs, gpt-5.5 writes a question needing both facts plus the ideal tool sequence.
# usage: python3 gen_traj.py   (reads ~/.oracle/oracle.db + ~/.oracle/azure.key; writes ~/.oracle/traj_data.jsonl)
import json, os, sqlite3, urllib.request, time, sys
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["ORACLE_LLM_KEY"]
URL = os.environ["ORACLE_LLM_URL"]
SYS = """You design retrieval training trajectories. For EACH numbered item (two related facts sharing an entity), write: a QUESTION whose answer needs BOTH facts, and the ideal tool sequence from: search(q), entity(name), graph(name), metric(name), STOP. Rules: 2-4 steps, each search query terse (3-6 words), the sequence must plausibly surface both facts. Return JSON {"items":[{"idx":int,"question":str,"steps":[{"tool":str,"arg":str}]}]}"""
def call(b):
    lines="\n".join(f"{i}: FACT_A: {a}\n   FACT_B: {bb}\n   SHARED_ENTITY: {e}" for i,(a,bb,e) in enumerate(b))
    body=json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":lines}],
        "response_format":{"type":"json_object"},"max_completion_tokens":12000,"reasoning_effort":"low"}).encode()
    for at in range(3):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"api-key":KEY,"Content-Type":"application/json"}),timeout=240))
            return json.loads(r["choices"][0]["message"]["content"]).get("items",[])
        except Exception as ex:
            if at==2: print("fail",ex,file=sys.stderr); return []
            time.sleep(5*(at+1))
db=sqlite3.connect(os.path.expanduser('~/.oracle/oracle.db'))
pairs=db.execute("""SELECT f1.statement, f2.statement, e.display FROM fact_entities a
  JOIN fact_entities b ON a.entity_id=b.entity_id AND a.fact_id < b.fact_id
  JOIN facts f1 ON f1.id=a.fact_id JOIN facts f2 ON f2.id=b.fact_id
  JOIN entities e ON e.id=a.entity_id
  WHERE f1.superseded_at IS NULL AND f2.superseded_at IS NULL AND e.seen_count BETWEEN 5 AND 200
  ORDER BY RANDOM() LIMIT 8000""").fetchall()
out=open(os.path.expanduser('~/.oracle/traj_data.jsonl'),'w')
B=6; batches=[pairs[i:i+B] for i in range(0,len(pairs),B)]
done=0
def work(b): return b, call(b)
with ThreadPoolExecutor(max_workers=8) as ex:
    for batch, items in ex.map(work, batches):
        by={it.get("idx"):it for it in items if isinstance(it,dict)}
        for i,(fa,fb,e) in enumerate(batch):
            it=by.get(i)
            if it and it.get("question") and it.get("steps"):
                out.write(json.dumps({"question":it["question"],"steps":it["steps"],"gold":[fa,fb],"entity":e})+"\n")
        done+=1
        if done%100==0: print(f"{done}/{len(batches)}",flush=True)
print("TRAJ_DONE")
