import json, sqlite3, re, os, urllib.request
import numpy as np
probes=[]
for path in ['probes_1k.tsv','probes.tsv']:
    for line in open(path):
        p=line.rstrip('\n').split('\t')
        if len(p)>=2 and p[0] and (len(p)<4 or not p[3]): probes.append((p[0],p[1]))
db=sqlite3.connect('oracle.db')
live={r[0]:r[1] for r in db.execute("SELECT id,statement FROM facts WHERE superseded_at IS NULL")}
lids,lV=[],[]
for line in open('local_vecs.jsonl'):
    d=json.loads(line)
    if d["id"] in live: lids.append(d["id"]); lV.append(d["v"])
lV=np.array(lV,dtype=np.float32)
lq={json.loads(l)["q"]:np.array(json.loads(l)["v"],dtype=np.float32) for l in open('question_vecs.jsonl')}
aids,aV=[],[]
for fid,blob in db.execute("SELECT v.fact_id,v.vec FROM fact_vecs v JOIN facts f ON f.id=v.fact_id WHERE f.superseded_at IS NULL"):
    aids.append(fid); aV.append(np.frombuffer(blob,dtype=np.float32))
aV=np.stack(aV)
key=os.environ["ORACLE_EMBED_KEY"]
def remote_embed(texts):
    body=json.dumps({"input":texts,"dimensions":512}).encode()
    req=urllib.request.Request(os.environ["ORACLE_EMBED_URL"],
        data=body,headers={"api-key":key,"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req,timeout=120))
    out=[None]*len(texts)
    for d in r["data"]:
        v=np.array(d["embedding"],dtype=np.float32); out[d["index"]]=v/np.linalg.norm(v)
    return out
qs=[q for q,_ in probes]; aq={}
for i in range(0,len(qs),32):
    for q,v in zip(qs[i:i+32],remote_embed(qs[i:i+32])): aq[q]=v
def hit5(qmap,ids,V):
    h=n=0
    for q,rx in probes:
        if q not in qmap: continue
        n+=1
        top=np.argsort(-(V@qmap[q]))[:5]
        try: pat=re.compile(rx,re.I)
        except: continue
        if any(pat.search(live[ids[t]]) for t in top): h+=1
    return h,n
lh,ln=hit5(lq,lids,lV); ah,an=hit5(aq,aids,aV)
print(f"LOCAL v2 embedder cosine-arm hit@5: {lh}/{ln} = {lh/ln:.3f}")
print(f"AZURE embedder    cosine-arm hit@5: {ah}/{an} = {ah/an:.3f}")
