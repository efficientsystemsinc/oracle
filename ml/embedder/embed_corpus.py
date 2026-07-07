"""Load trained embedder params, embed all live facts + emit vecs jsonl.

usage: python3 embed_corpus.py   (run in a dir with embedder.msgpack + live_facts.jsonl; writes local_vecs.jsonl)
"""
import json, numpy as np, jax, jax.numpy as jnp
from flax import serialization
from transformers import AutoTokenizer, FlaxAutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = FlaxAutoModel.from_pretrained("bert-base-uncased", from_pt=True)
params = {"enc": enc.params, "proj": jnp.zeros((768, 512))}
params = serialization.from_bytes(params, open("embedder.msgpack","rb").read())

@jax.jit
def embed(ids, mask):
    out = enc(input_ids=ids, attention_mask=mask, params=params["enc"])[0]
    m = mask[..., None]
    pooled = (out * m).sum(1) / m.sum(1)
    v = pooled @ params["proj"]
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

ids, texts = [], []
for line in open("live_facts.jsonl"):
    d = json.loads(line); ids.append(d["id"]); texts.append(d["statement"])
out = open("local_vecs.jsonl", "w")
B = 256
for i in range(0, len(texts), B):
    e = tok(texts[i:i+B], truncation=True, max_length=128, padding="max_length", return_tensors="np")
    v = np.array(embed(e["input_ids"], e["attention_mask"]))
    for j, fid in enumerate(ids[i:i+B]):
        out.write(json.dumps({"id": fid, "v": [round(float(x),6) for x in v[j]]})+"\n")
    if i % 5120 == 0: print(i, flush=True)
print("DONE", len(ids))
