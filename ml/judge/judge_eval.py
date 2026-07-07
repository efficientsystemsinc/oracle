"""Eval-only: load trained judge params, score any test jsonl.

usage: python3 judge_eval.py <model.msgpack> <test.jsonl>   (imports render/load from judge_train.py; prints acc + per-class P/R)
"""
import sys, json
import numpy as np, jax
from flax import serialization
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
sys.argv0, model_path, test_path = sys.argv
LABELS=["UPHOLD","REOPEN","REOPEN_CONTRADICT"]; L2I={l:i for i,l in enumerate(LABELS)}
from judge_train import render, load  # reuse rendering
tok=AutoTokenizer.from_pretrained("bert-base-uncased")
model=FlaxAutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=3,from_pt=True)
params=serialization.from_bytes(model.params, open(model_path,'rb').read())
te,y=load(test_path,tok,256)
@jax.jit
def pred(ids,mask,tt): return model(input_ids=ids,attention_mask=mask,token_type_ids=tt,params=params,train=False)[0].argmax(-1)
ps=[]
for i in range(0,len(y),64):
    sl=slice(i,i+64)
    ps.append(np.array(pred(te["input_ids"][sl],te["attention_mask"][sl],te["token_type_ids"][sl])))
p=np.concatenate(ps)
acc=(p==y).mean(); yb=(y!=0).astype(int); pb=(p!=0).astype(int)
per={}
for c,name in enumerate(LABELS):
    tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()
    per[name]=(round(float(tp/max(tp+fp,1)),3), round(float(tp/max(tp+fn,1)),3), int((y==c).sum()))
print(f"{test_path}: n={len(y)} acc={acc:.4f} binary={float((yb==pb).mean()):.4f} per-class={per}")
