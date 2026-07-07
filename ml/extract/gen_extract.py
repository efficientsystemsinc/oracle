# Distill the extraction teacher: run every transcript chunk through gpt-5.5 (round-robin over 3 Azure deployments) with the extract prompt; resumable.
# usage: python3 gen_extract.py   (reads ~/.oracle/extract_chunks.jsonl + ~/.oracle/azure.key + ~/.oracle/azure_sweden.key + ./extract_prompt.txt; appends to ~/.oracle/extract_distill.jsonl)
import json, os, urllib.request, time, sys, datetime
from concurrent.futures import ThreadPoolExecutor
POOLS = [(os.environ["ORACLE_LLM_URL"], os.environ["ORACLE_LLM_KEY"])]
import itertools, threading
_rr = itertools.cycle(range(len(POOLS)))
_rrlock = threading.Lock()
SYS = open(os.path.dirname(os.path.abspath(__file__))+'/extract_prompt.txt').read()
def call(user):
    body=json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":user}],
        "response_format":{"type":"json_object"},"max_completion_tokens":28000,"reasoning_effort":"low"}).encode()
    for a in range(4):
        with _rrlock: i = next(_rr)
        url, key = POOLS[i]
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,data=body,headers={"api-key":key,"Content-Type":"application/json"}),timeout=280))
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if a==3: return None
            time.sleep(4*(a+1))
chunks=[json.loads(l) for l in open(os.path.expanduser('~/.oracle/extract_chunks.jsonl'))]
out=open(os.path.expanduser('~/.oracle/extract_distill.jsonl'),'a')
done_texts=set()
try:
    for l in open(os.path.expanduser('~/.oracle/extract_distill.jsonl')):
        done_texts.add(json.loads(l)["input"])
except FileNotFoundError: pass
work=[c for c in chunks if hash(("Repo: %s\n"%c["repo"])[:2000-1990]+c["text"][:1990]) or True]
todo=[c for c in chunks]
print("todo", len(todo), "already", len(done_texts), flush=True)
n=0
def job(c):
    date=datetime.datetime.utcfromtimestamp(c["event_time"]).strftime("%Y-%m-%d")
    user=f"Repo: {c['repo']}\nSession date: {date}\n\nTRANSCRIPT:\n{c['text']}"
    if user in done_texts: return None
    resp=call(user)
    return (user, resp)
with ThreadPoolExecutor(max_workers=64) as ex:
    for res in ex.map(job, todo):
        if res and res[1]:
            out.write(json.dumps({"input":res[0],"output":res[1]})+"\n"); out.flush()
        n+=1
        if n%100==0: print(f"{n}/{len(todo)}",flush=True)
print("EXTRACT_DISTILL_DONE")
