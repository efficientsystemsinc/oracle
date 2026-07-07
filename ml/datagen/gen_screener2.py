"""Historical chunks + LLM yield labels for the screener.

usage: python3 gen_screener2.py   (reads ~/.claude/projects/*/*.jsonl + ~/.oracle/azure.key; appends to ~/.oracle/screener_data.jsonl)
"""
import json, os, glob, random, urllib.request, time, sys
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["ORACLE_LLM_KEY"]
URL = os.environ["ORACLE_LLM_URL"]
SYS = """For EACH numbered transcript chunk, estimate how many DURABLE facts a memory extractor would keep (decisions, gotchas, infra truths, benchmark numbers, preferences — NOT transient debugging). Return JSON {"items":[{"idx":int,"yield":int}]} (yield 0-12)."""
def render(path, cap=8000):
    txt=[]
    total=0
    for line in open(path, errors='replace'):
        try: r=json.loads(line)
        except: continue
        t=r.get("type")
        if t not in ("user","assistant"): continue
        c=(r.get("message") or {}).get("content")
        if isinstance(c,str): s=c
        elif isinstance(c,list): s="\n".join(x.get("text","") for x in c if isinstance(x,dict) and x.get("type")=="text")
        else: continue
        if s.strip():
            txt.append(("USER: " if t=="user" else "ASSISTANT: ")+s[:1500])
            total+=len(s)
            if total>cap: break
    return "\n".join(txt)[:cap]
files=[f for f in glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl'))
       if '/-private-tmp' not in f and '/-tmp-' not in f]
random.seed(7); random.shuffle(files)
chunks=[]
for f in files[:2500]:
    c=render(f)
    if len(c)>500: chunks.append(c)
print("chunks", len(chunks), flush=True)
def call(b):
    lines="\n\n".join(f"=== CHUNK {i} ===\n{c}" for i,c in enumerate(b))
    body=json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":lines}],
        "response_format":{"type":"json_object"},"max_completion_tokens":4000,"reasoning_effort":"low"}).encode()
    for a in range(3):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"api-key":KEY,"Content-Type":"application/json"}),timeout=240))
            return json.loads(r["choices"][0]["message"]["content"]).get("items",[])
        except Exception as e:
            if a==2: return []
            time.sleep(5*(a+1))
out=open(os.path.expanduser('~/.oracle/screener_data.jsonl'),'a')
B=4; batches=[chunks[i:i+B] for i in range(0,len(chunks),B)]
done=0
def work(b): return b, call(b)
with ThreadPoolExecutor(max_workers=8) as ex:
    for batch, items in ex.map(work, batches):
        by={it.get("idx"):it for it in items if isinstance(it,dict)}
        for i,c in enumerate(batch):
            it=by.get(i)
            if it is not None and isinstance(it.get("yield"), int):
                out.write(json.dumps({"text":c,"yield":it["yield"],"src":"llm-label"})+"\n")
        done+=1
        if done%50==0: print(f"{done}/{len(batches)}",flush=True)
print("SCREENER_DONE")
