# Embed probe questions with the trained local embedder; emits question vectors for ab_eval.py.
# usage: python3 embed_questions.py   (run in a dir with embedder.msgpack + probe_questions.jsonl; writes question_vecs.jsonl)
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
    m = mask[..., None]; pooled = (out*m).sum(1)/m.sum(1)
    v = pooled @ params["proj"]
    return v/(jnp.linalg.norm(v,axis=-1,keepdims=True)+1e-9)
qs=[json.loads(l)["q"] for l in open("probe_questions.jsonl")]
out=open("question_vecs.jsonl","w")
for i in range(0,len(qs),256):
    e=tok(qs[i:i+256],truncation=True,max_length=128,padding="max_length",return_tensors="np")
    v=np.array(embed(e["input_ids"],e["attention_mask"]))
    for j,q in enumerate(qs[i:i+256]):
        out.write(json.dumps({"q":q,"v":[round(float(x),6) for x in v[j]]})+"\n")
print("QDONE",len(qs))
