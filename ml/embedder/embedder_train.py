"""oracle query embedder: contrastive fine-tune (InfoNCE, in-batch negatives)
on (statement, paraphrase) pairs so a local ~110M encoder replaces the Azure
embedding call. Mean-pooled bert-base -> 512-d projection, temperature 0.05.
Eval: para->statement retrieval recall@10 on held-out pairs.

usage: python3 embedder_train.py [--model M] [--epochs N] [--batch B] [--lr LR] [--max-len L] [--dim D] [--out NAME]   (reads embed_pairs.jsonl in cwd; writes <out>.msgpack)
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state
from transformers import AutoTokenizer, FlaxAutoModel

def load_pairs(path):
    A, B = [], []
    for line in open(path):
        d = json.loads(line)
        A.append(d["statement"]); B.append(d["paraphrase"])
    return A, B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--out", default="embedder")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    enc_model = FlaxAutoModel.from_pretrained(a.model, from_pt=True)

    A, B = load_pairs("embed_pairs.jsonl")
    n = len(A); cut = int(n * 0.95)
    print("pairs", n, "train", cut)

    def encode_np(texts):
        return tok(texts, truncation=True, max_length=a.max_len,
                   padding="max_length", return_tensors="np")

    trA, trB = encode_np(A[:cut]), encode_np(B[:cut])
    teA, teB = encode_np(A[cut:]), encode_np(B[cut:])

    # projection head params
    rng = jax.random.PRNGKey(7)
    W = jax.random.normal(rng, (768, a.dim)) * 0.02
    params = {"enc": enc_model.params, "proj": W}

    def embed(params, ids, mask):
        out = enc_model(input_ids=ids, attention_mask=mask, params=params["enc"])[0]
        m = mask[..., None]
        pooled = (out * m).sum(1) / m.sum(1)
        v = pooled @ params["proj"]
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

    steps = (cut // a.batch) * a.epochs
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, int(steps * 0.1), steps)
    tx = optax.adamw(sched, weight_decay=0.01)
    state = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

    @jax.jit
    def step(state, ia, ma, ib, mb):
        def loss_fn(p):
            va, vb = embed(p, ia, ma), embed(p, ib, mb)
            logits = va @ vb.T / 0.05
            y = jnp.arange(va.shape[0])
            return (optax.softmax_cross_entropy_with_integer_labels(logits, y).mean() +
                    optax.softmax_cross_entropy_with_integer_labels(logits.T, y).mean()) / 2
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def embed_j(params, ids, mask):
        return embed(params, ids, mask)

    def recall_at(params, k=10):
        def all_vecs(e):
            vs = []
            for i in range(0, len(e["input_ids"]), 256):
                sl = slice(i, i + 256)
                vs.append(np.array(embed_j(params, e["input_ids"][sl], e["attention_mask"][sl])))
            return np.concatenate(vs)
        va, vb = all_vecs(teA), all_vecs(teB)
        sims = vb @ va.T  # paraphrase -> statement
        hits = sum(1 for i in range(len(vb)) if i in np.argsort(-sims[i])[:k])
        return hits / len(vb)

    rng2 = np.random.default_rng(7)
    for ep in range(a.epochs):
        idx = rng2.permutation(cut)
        t0, tot, ns = time.time(), 0.0, 0
        for s in range(cut // a.batch):
            sl = idx[s * a.batch:(s + 1) * a.batch]
            state, loss = step(state, trA["input_ids"][sl], trA["attention_mask"][sl],
                               trB["input_ids"][sl], trB["attention_mask"][sl])
            tot += float(loss); ns += 1
        r10 = recall_at(state.params)
        print(f"epoch {ep}: loss {tot/ns:.4f} heldout para->stmt recall@10 {r10:.4f} ({time.time()-t0:.0f}s)")

    with open(f"{a.out}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    print("SAVED", a.out)

if __name__ == "__main__":
    main()
