#!/usr/bin/env python3
"""Export bert weights from oracle's ONNX models to .safetensors for the MLX engine.

Why ONNX as the source (not HF/torch): the ONNX graphs are the exact weights the
production ORT path serves — including the embedder's 768x512 projection, which
exists ONLY in the ONNX graph (baked in at export time). Reading the ONNX
initializers guarantees bit-identical fp32 weights with zero re-export drift and
needs no torch/transformers install (just `onnx safetensors numpy` in the
scripts/.venv-mlx py3.13 venv).

Layout convention in the output file: every 2D linear weight is stored (in, out)
so the C++ side computes plain `x @ W + b` with no transposes.

Usage:
  scripts/.venv-mlx/bin/python scripts/export_safetensors.py [models_dir]
Writes <models_dir>/{judge_v2_onnx,embedder_v3_onnx}/mlx_weights.safetensors
"""

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

H, FFN = 768, 3072
LAYERS = 12
# per-layer weight-MatMul order in both graphs (topological):
SLOT_NAMES = ["q", "k", "v", "attn_out", "ffn_in", "ffn_out"]
SLOT_SHAPES = [(H, H), (H, H), (H, H), (H, H), (H, FFN), (FFN, H)]


def fail(msg):
    raise SystemExit(f"export_safetensors: FATAL: {msg}")


def export(model_dir, kind):
    path = os.path.join(model_dir, "model.onnx")
    m = onnx.load(path)  # loads model.onnx.data external tensors too
    inits = {i.name: i for i in m.graph.initializer}

    def get(name):
        if name not in inits:
            fail(f"{path}: missing initializer {name!r}")
        return numpy_helper.to_array(inits[name]).astype(np.float32)

    out = {}
    # --- embeddings + all HF-named tensors (biases, LN, embeddings, heads) ---
    out["emb.word"] = get("bert.embeddings.word_embeddings.weight")
    out["emb.pos"] = get("bert.embeddings.position_embeddings.weight")
    out["emb.type"] = get("bert.embeddings.token_type_embeddings.weight")
    out["emb.ln.w"] = get("bert.embeddings.LayerNorm.weight")
    out["emb.ln.b"] = get("bert.embeddings.LayerNorm.bias")
    for i in range(LAYERS):
        p = f"bert.encoder.layer.{i}"
        out[f"L{i}.q.b"] = get(f"{p}.attention.self.query.bias")
        out[f"L{i}.k.b"] = get(f"{p}.attention.self.key.bias")
        out[f"L{i}.v.b"] = get(f"{p}.attention.self.value.bias")
        out[f"L{i}.attn_out.b"] = get(f"{p}.attention.output.dense.bias")
        out[f"L{i}.attn_ln.w"] = get(f"{p}.attention.output.LayerNorm.weight")
        out[f"L{i}.attn_ln.b"] = get(f"{p}.attention.output.LayerNorm.bias")
        out[f"L{i}.ffn_in.b"] = get(f"{p}.intermediate.dense.bias")
        out[f"L{i}.ffn_out.b"] = get(f"{p}.output.dense.bias")
        out[f"L{i}.out_ln.w"] = get(f"{p}.output.LayerNorm.weight")
        out[f"L{i}.out_ln.b"] = get(f"{p}.output.LayerNorm.bias")

    # --- anonymous MatMul weights, recovered by topological order ---
    # A "weight MatMul" is a MatMul whose 2nd input is an initializer.
    wmm = [
        n.input[1]
        for n in m.graph.node
        if n.op_type == "MatMul" and n.input[1] in inits
    ]
    if kind == "embedder":
        if wmm[-1] != "proj":
            fail(f"expected trailing proj MatMul, got {wmm[-1]!r}")
        proj = get("proj")
        if proj.shape != (H, 512):
            fail(f"proj shape {proj.shape}")
        out["proj"] = proj
        wmm = wmm[:-1]
    if len(wmm) != LAYERS * 6:
        fail(f"{path}: found {len(wmm)} weight MatMuls, want {LAYERS * 6}")
    for i in range(LAYERS):
        for s, (slot, shape) in enumerate(zip(SLOT_NAMES, SLOT_SHAPES)):
            w = get(wmm[i * 6 + s])
            if w.shape != shape:
                fail(f"layer {i} slot {slot}: shape {w.shape}, want {shape}")
            out[f"L{i}.{slot}.w"] = w  # already (in, out) in ONNX MatMul form

    # --- heads ---
    if kind == "judge":
        # Gemm weights are HF (out, in); transpose to (in, out).
        out["pooler.w"] = get("bert.pooler.dense.weight").T.copy()
        out["pooler.b"] = get("bert.pooler.dense.bias")
        out["cls.w"] = get("classifier.weight").T.copy()
        out["cls.b"] = get("classifier.bias")
        if out["cls.w"].shape != (H, 3):
            fail(f"classifier shape {out['cls.w'].shape}")

    dst = os.path.join(model_dir, "mlx_weights.safetensors")
    save_file(out, dst)
    total = sum(v.nbytes for v in out.values())
    print(f"{dst}: {len(out)} tensors, {total / 1e6:.1f} MB fp32")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.oracle/models")
    export(os.path.join(root, "judge_v2_onnx"), "judge")
    export(os.path.join(root, "embedder_v3_onnx"), "embedder")


if __name__ == "__main__":
    main()
