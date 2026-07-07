"""oracle judge distillation: 3-class supersession verdict (UPHOLD / REOPEN /
REOPEN_CONTRADICT) from fact pairs + dates + evidence tiers.
Teacher labels: gpt-5.5 skeptic (repair_done). Runs on TPU v6e via flax.
Usage: python3 judge_train.py --model bert-base-uncased --epochs 3
"""
import argparse, json, time
from collections import Counter
from datetime import datetime

import jax, jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import serialization
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

LABELS = ["UPHOLD", "REOPEN", "REOPEN_CONTRADICT"]
L2I = {l: i for i, l in enumerate(LABELS)}


def render(d):
    def side(p):
        date = datetime.utcfromtimestamp(d[p + "_date"]).strftime("%Y-%m-%d")
        return f"[{date}] [{d[p+'_ev']}] [{d[p+'_kind']}] {d[p]}"
    delta = round((d["new_date"] - d["old_date"]) / 86400)
    return f"OLD {side('old')}", f"NEW (+{delta}d) {side('new')}"


def load(path, tok, max_len):
    X, Y = [], []
    A, B = [], []
    for line in open(path):
        d = json.loads(line)
        a, b = render(d)
        A.append(a); B.append(b); Y.append(L2I[d["label"]])
    enc = tok(A, B, truncation=True, max_length=max_len, padding="max_length", return_tensors="np")
    return enc, np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="judge_model")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=3, from_pt=True)

    tr, ytr = load("judge_train.jsonl", tok, a.max_len)
    te, yte = load("judge_test.jsonl", tok, a.max_len)
    print("train", len(ytr), Counter(ytr.tolist()), "test", len(yte))

    # inverse-frequency class weights (REOPEN_CONTRADICT is 5%)
    freq = np.bincount(ytr, minlength=3).astype(np.float32)
    w = (freq.sum() / (3 * freq)); w = w / w.mean()
    cw = jnp.array(w)
    print("class weights", w)

    steps_per_epoch = len(ytr) // a.batch
    total = steps_per_epoch * a.epochs
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, int(total * 0.1), total)
    tx = optax.adamw(sched, weight_decay=0.01)
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx)

    @jax.jit
    def step(state, ids, mask, ttype, y):
        def loss_fn(params):
            logits = state.apply_fn(input_ids=ids, attention_mask=mask,
                                    token_type_ids=ttype, params=params, train=False)[0]
            onehot = jax.nn.one_hot(y, 3)
            ls = optax.softmax_cross_entropy(logits, onehot) * cw[y]
            return ls.mean(), logits
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def predict(params, ids, mask, ttype):
        return state.apply_fn(input_ids=ids, attention_mask=mask,
                              token_type_ids=ttype, params=params, train=False)[0].argmax(-1)

    def evaluate(params):
        preds = []
        for i in range(0, len(yte), a.batch):
            sl = slice(i, i + a.batch)
            preds.append(np.array(predict(params, te["input_ids"][sl],
                                          te["attention_mask"][sl], te["token_type_ids"][sl])))
        p = np.concatenate(preds)
        acc = (p == yte).mean()
        per = {}
        for c, name in enumerate(LABELS):
            tp = ((p == c) & (yte == c)).sum(); fp = ((p == c) & (yte != c)).sum()
            fn = ((p != c) & (yte == c)).sum()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            per[name] = (round(float(prec), 3), round(float(rec), 3), int((yte == c).sum()))
        return acc, per, p

    rng = np.random.default_rng(7)
    best = 0.0
    for ep in range(a.epochs):
        idx = rng.permutation(len(ytr))
        t0, tot = time.time(), 0.0
        for s in range(steps_per_epoch):
            sl = idx[s * a.batch:(s + 1) * a.batch]
            state, loss = step(state, tr["input_ids"][sl], tr["attention_mask"][sl],
                               tr["token_type_ids"][sl], ytr[sl])
            tot += float(loss)
        acc, per, _ = evaluate(state.params)
        print(f"epoch {ep}: loss {tot/steps_per_epoch:.4f} test-acc {acc:.4f} "
              f"per-class(P,R,n) {per} ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc
            with open(f"{a.out}.msgpack", "wb") as f:
                f.write(serialization.to_bytes(state.params))
    print("BEST", round(float(best), 4))
    # binary view: was the original supersession wrong at all? (REOPEN* vs UPHOLD)
    acc, per, p = evaluate(state.params)
    yb = (yte != 0).astype(int); pb = (p != 0).astype(int)
    print("binary reopen-vs-uphold acc", round(float((yb == pb).mean()), 4))


if __name__ == "__main__":
    main()
