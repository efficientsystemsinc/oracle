# Generate (raw transcript excerpt -> clean fact) rewrite training pairs: real fact quotes from the oracle DB plus gpt-5.5-synthesized excerpts for facts without one.
# usage: python3 gen_rewrite.py   (reads ~/.oracle/oracle.db + ~/.oracle/azure.key; writes ~/.oracle/rewrite_data.jsonl)
import json, os, sqlite3, urllib.request, time, sys
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["ORACLE_LLM_KEY"]
URL = os.environ["ORACLE_LLM_URL"]
SYS = """For EACH numbered fact, write a realistic RAW transcript excerpt (3-8 lines) from an AI coding session in which this fact becomes evident — tool output, agent narration, user remarks; messy, with surrounding noise, the fact NOT stated cleanly. Then the training pair is (your excerpt -> the clean fact). Return JSON {"items":[{"idx":int,"excerpt":str}]}"""
def call(b):
    lines="\n".join(f"{i}: {s}" for i,s in enumerate(b))
    body=json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":lines}],
        "response_format":{"type":"json_object"},"max_completion_tokens":14000,"reasoning_effort":"low"}).encode()
    for a in range(3):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"api-key":KEY,"Content-Type":"application/json"}),timeout=240))
            return json.loads(r["choices"][0]["message"]["content"]).get("items",[])
        except Exception as e:
            if a==2: print("fail",e,file=sys.stderr); return []
            time.sleep(5*(a+1))
db=sqlite3.connect(os.path.expanduser('~/.oracle/oracle.db'))
# real quotes first (they're gold), then synthetic for the rest
out=open(os.path.expanduser('~/.oracle/rewrite_data.jsonl'),'w')
nreal=0
for s,q in db.execute("SELECT statement, quote FROM facts WHERE quote IS NOT NULL AND length(quote)>40"):
    out.write(json.dumps({"excerpt":q,"fact":s,"src":"real"})+"\n"); nreal+=1
rows=[r[0] for r in db.execute("SELECT statement FROM facts WHERE superseded_at IS NULL AND (quote IS NULL OR length(quote)<=40) ORDER BY RANDOM() LIMIT 12000")]
B=8; batches=[rows[i:i+B] for i in range(0,len(rows),B)]
done=0
def work(b): return b, call(b)
with ThreadPoolExecutor(max_workers=8) as ex:
    for batch, items in ex.map(work, batches):
        by={it.get("idx"):it for it in items if isinstance(it,dict)}
        for i,s in enumerate(batch):
            it=by.get(i)
            if it and len((it.get("excerpt") or ""))>60:
                out.write(json.dumps({"excerpt":it["excerpt"],"fact":s,"src":"synth"})+"\n")
        done+=1
        if done%100==0: print(f"{done}/{len(batches)}",flush=True)
print("REWRITE_DONE real=",nreal)
