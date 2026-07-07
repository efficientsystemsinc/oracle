"""Export the trained oracle query embedder (flax msgpack) to ONNX.

Pipeline: intfloat/e5-base-v2 BERT encoder + 768x512 projection,
masked mean-pool over last_hidden_state, L2-normalized 512-d output.

Loads ~/.oracle/models/embedder_v3.msgpack ({"enc": ..., "proj": ...} pytree),
rebuilds the encoder as a PyTorch BertModel (from_flax), wraps encoder +
pool + proj + normalize in one nn.Module, exports to ONNX (opset 17,
dynamic batch, output name "embedding"), verifies flax<->onnx parity on
sample texts (max abs diff must be < 1e-4), and saves model + tokenizer
to ~/.oracle/models/embedder_v3_onnx/.

Run inside a venv with: transformers==4.46.3 torch flax jax[cpu] onnx onnxruntime onnxscript

usage: python3 export_embedder_onnx.py   (reads ~/.oracle/models/embedder_v3.msgpack; writes ~/.oracle/models/embedder_v3_onnx/)
"""
import os
import sys
import tempfile

import numpy as np

BASE_MODEL = "intfloat/e5-base-v2"
MSGPACK = os.path.expanduser("~/.oracle/models/embedder_v3.msgpack")
OUT_DIR = os.path.expanduser("~/.oracle/models/embedder_v3_onnx")
MAX_LEN = 128
DIM = 512
PARITY_TOL = 1e-4

SAMPLES = [
    "the quasar backing store moved to postgres 16 on 192.0.2.10 behind pgbouncer",
    "meadow runner SSH access always uses user ci-user with the ed25519 key",
    "query: how do I redeploy the lighthouse API box after a wedge",
    "bramble refreshes the feature-flag page every two hours",
    "short",
]


def main() -> None:
    import jax.numpy as jnp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from flax import serialization
    from transformers import AutoTokenizer, BertModel, FlaxBertModel

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    # --- restore trained flax params -------------------------------------
    flax_enc = FlaxBertModel.from_pretrained(BASE_MODEL, from_pt=True)
    template = {"enc": flax_enc.params, "proj": jnp.zeros((768, DIM))}
    with open(MSGPACK, "rb") as f:
        params = serialization.from_bytes(template, f.read())
    flax_enc.params = params["enc"]
    proj = np.asarray(params["proj"], dtype=np.float32)
    assert proj.shape == (768, DIM), proj.shape

    def flax_forward(texts):
        e = tok(texts, truncation=True, max_length=MAX_LEN,
                padding="max_length", return_tensors="np")
        out = flax_enc(input_ids=e["input_ids"],
                       attention_mask=e["attention_mask"])[0]
        m = e["attention_mask"][..., None]
        pooled = (out * m).sum(1) / m.sum(1)
        v = pooled @ params["proj"]
        return np.asarray(v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9))

    # --- flax encoder -> torch BertModel ----------------------------------
    with tempfile.TemporaryDirectory() as td:
        flax_enc.save_pretrained(td)
        pt_bert = BertModel.from_pretrained(td, from_flax=True, add_pooling_layer=False)
    pt_bert.eval()

    class Embedder(nn.Module):
        def __init__(self, bert: BertModel, proj_w: np.ndarray):
            super().__init__()
            self.bert = bert
            self.proj = nn.Parameter(torch.from_numpy(proj_w.copy()))

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(out.dtype)
            pooled = (out * m).sum(1) / m.sum(1)
            return F.normalize(pooled @ self.proj, dim=-1)

    model = Embedder(pt_bert, proj).eval()

    # --- export ------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    onnx_path = os.path.join(OUT_DIR, "model.onnx")
    dummy = tok(["dummy text"], truncation=True, max_length=MAX_LEN,
                padding="max_length", return_tensors="pt")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    tok.save_pretrained(OUT_DIR)

    # --- parity check -------------------------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    enc = tok(SAMPLES, truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="np")
    onnx_out = sess.run(["embedding"], {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    })[0]
    flax_out = flax_forward(SAMPLES)
    diff = float(np.max(np.abs(onnx_out - flax_out)))
    print(f"parity max abs diff (flax vs onnx, {len(SAMPLES)} texts): {diff:.3e}")
    if diff >= PARITY_TOL:
        print(f"FAIL: parity {diff:.3e} >= {PARITY_TOL}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: exported to {OUT_DIR} (model.onnx + tokenizer), dim={DIM}, max_len={MAX_LEN}")


if __name__ == "__main__":
    main()
