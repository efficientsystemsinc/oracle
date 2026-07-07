# MLX performance notes for cpp/oraclemlx (BERT embed + brute-force cosine top-k, M4 Max)

Researched 2026-07-07. Target: C++ MLX engine doing ~110M-param BERT-class encoding
plus fused matmul→top-k over ~50k x 512 fp16 corpus vectors. Every nontrivial claim cited.

## 1. Custom Metal kernels (`mx.fast.metal_kernel` / `mx::fast::metal_kernel`)

- API: you supply only the kernel **body**; MLX generates the signature from
  `input_names` / `output_names` and auto-adds shape/strides/ndim buffers plus Metal
  attributes (`thread_position_in_grid` etc.) it sees referenced in the source.
  Template params (`template=[("T", mx.float32)]`) accept Dtype/int/bool. Same machinery
  exists in C++ (`mlx/fast.h`).
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
- `ensure_row_contiguous=True` (default) copies non-contiguous inputs before launch.
  Disable it and index with `elem_to_loc()` + passed strides to avoid the copy.
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- **Build the kernel object once, call it many times** — construction implies Metal
  library creation + JIT compile. Repeated per-call overhead is real: issue #1828 reports
  large-loop (>10k calls) workloads slower than raw py-metal-compute, dominated by lazy-graph
  build + host readback, not kernel time. Batch work per dispatch and keep results in MLX
  arrays as long as possible.
  https://github.com/ml-explore/mlx/issues/1828
- `atomic_outputs=True` + `init_value=0` for scatter/reduce-style outputs.
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- When custom beats composed ops: **fusion that cuts memory traffic / dispatch count**.
  Docs' grid_sample example: 8x forward, 40x backward vs composed ops; a fused quantize
  pipeline reported 2.7x from eliminating kernel-launch stalls. For elementwise chains,
  prefer `mx.compile` first (it fuses on GPU); reach for a hand kernel only when the
  pattern (e.g. matmul-epilogue + reduction like fused score+top-k) can't be expressed.
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  https://github.com/manishklach/mlx-metal-kernels (community fast attention/decode kernels)
  https://github.com/ssmall256/mlx-metal-kernels-skill (examples + optimization patterns)
- Threadgroup dims must each be ≤ the corresponding grid dim; grid = total threads
  (MLX uses non-uniform dispatch).
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

## 2. Matmul + top-k for cosine search (~50k x 512 fp16)

- Standard pattern: pre-normalize corpus once; per query `scores = q @ C.T` then
  `mx.argpartition(-scores, kth=k)[:k]` and a tiny sort of the k winners.
  `x @ W.T` is faster than `x @ W` in MLX (better GEMM layout) — store the corpus so the
  contraction is over the last axis of both, i.e. keep C as (N, D) and do `q @ C.T`.
  https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
- `mx.argpartition` **does run on GPU** (Metal backend implements sort/partition; it is
  used exactly this way in MLX-native kNN work: NNDescent on Apple Silicon uses
  `mx.argpartition` to avoid full sorts, distances via GPU matmul).
  https://arxiv.org/html/2603.04035v2 (mlx-vis)
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html
- Cost intuition at this scale: the GEMM is ~50k×512 ≈ 26 MFLOP·2/query — trivially fast;
  the selection over 50k fp32 scores is bandwidth-trivial (~200 KB). The dominant costs are
  **dispatch/graph overhead and host readback**, not math. So: batch queries (B×512 @
  512×50k), do argpartition on-device, and only `mx.eval`/read back the k indices+scores.
  Avoid `mx.eval` per query — evaluate once per batch (lazy-eval batching guidance).
  https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
- MLX's sort family is bitonic/partition-based on Metal; for a single top-k over 50k rows
  full `argsort` is also acceptable, but argpartition avoids the O(n log n) pass —
  measured wins reported in MLX kNN pipelines.
  https://arxiv.org/html/2603.04035v2
- A fused "GEMM epilogue keeps only running top-k" custom kernel is the theoretical
  optimum (see RadiK-style GPU top-k literature), but at 50k×512 the composed
  matmul+argpartition is already ~memory-bound-trivial; fuse only if profiling shows
  dispatch overhead dominating. https://arxiv.org/pdf/2501.14336
- Prior art: Faiss-mlx, a Metal-accelerated vector search lib on MLX — worth skimming for
  their brute-force flat-index kernel choices. https://github.com/MLXPorts/Faiss-mlx
- Gotcha for repeated identical queries in evals/tests: MLX is lazy — build the whole
  score→partition→take graph and eval once; interleaving numpy conversions per step is
  the #1828 failure mode. https://github.com/ml-explore/mlx/issues/1828

## 3. Quantization (mx.quantize / quantized_matmul)

- `mx.quantize(w, group_size, bits, mode)`: affine mode supports group_size {32, 64
  (default), 128} and bits {2,3,4 (default),5,6,8}; also mxfp4/mxfp8 (gs=32) and nvfp4
  (gs=16). Returns (w_q packed in uint32, scales, biases). Last dim must divide group_size;
  w must be ≥2-D. `mx.quantized_matmul(x, w_q, scales, biases, transpose=True, ...)` runs
  the fused dequant-GEMM. https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
- Smaller group_size → better accuracy, slightly more memory/scale traffic.
  https://deepwiki.com/ml-explore/mlx/7-quantization
- Speed: quantized_matmul is a **memory-bandwidth win** — big for decode-style GEMV, more
  modest for compute-bound prefill/encoder GEMMs where activations are fp16 anyway. On
  M-series, 4-bit gs=64 is the commonly recommended speed/quality point; ~97% task-score
  retention at ~3.8x memory reduction reported for LLMs.
  https://branch8.com/posts/apple-silicon-mlx-llm-inference-optimization-tutorial
- **BERT-class (~110M) encoders are more quantization-sensitive than big decoders.**
  Transformer-quantization studies show BERT activations (esp. residual/GELU ranges) are
  the hard part; weight-only 4-bit is generally a reasonable accuracy/compression
  trade-off, but 4-bit W+A needs special treatment (MKQ-BERT).
  https://arxiv.org/pdf/2109.12948 https://arxiv.org/pdf/2203.13483
- Practical recommendation for oraclemlx: a 110M encoder is ~220 MB fp16 — memory isn't
  the constraint on M4 Max. Prefer **fp16 weights (or 8-bit affine, gs=64) for the encoder**
  and validate embedding quality (cosine-sim regression vs fp32 on a held-out set) before
  going 4-bit; embedding quality degrades before classification accuracy does. If 4-bit,
  use gs=32 and keep embeddings/LayerNorm unquantized (MLX quantizes only Linear/Embedding
  via nn.quantize; norms stay fp).
  https://deepwiki.com/ml-explore/mlx/7-quantization
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html

## 4. Unified memory / zero-copy / compile

- Unified memory model: arrays live in shared CPU/GPU memory; you pick the device per-op
  (`stream=mx.gpu`), never move data. CPU-op ↔ GPU-op interleave with zero copies.
  https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- **Wrapping existing host buffers**: historically `mlx::array(ptr, shape, dtype)` copies.
  Issue #2855 requested a no-copy constructor (allocator::Buffer + deleter); it was closed
  via PR #2875 (Dec 2025) — check your pinned MLX version for the no-copy C++ init before
  relying on it; otherwise pay one memcpy at load and keep everything in MLX arrays after.
  https://github.com/ml-explore/mlx/issues/2855
  https://github.com/ml-explore/mlx/pull/2875
- Cheapest load path for a big corpus matrix: `mx.load` of an .npy/.safetensors is lazy —
  cast/reshape **before** first eval so you never materialize the fp32 original (cuts peak
  memory ~1/3 when downcasting weights). Same trick applies to model weights.
  https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
- Donation: MLX donates input buffers to outputs automatically when refcount allows —
  keep temporaries tightly scoped and drop references before `mx.eval` so buffers get
  reused; MLX also caches freed buffers (watch cache growth under variable shapes).
  https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
- `mx.compile`: traces and fuses (esp. elementwise) into fewer kernels. Recompiles on
  **shape change** or changed captured constants — for variable-length BERT batches either
  bucket sequence lengths to a few fixed shapes, or use `shapeless=True` (careful: any
  `reshape` baked from the first trace can silently misbehave).
  https://ml-explore.github.io/mlx/build/html/usage/compile.html
- Pass closed-over arrays via `inputs=`/`outputs=` (Python) so the compiled graph doesn't
  swallow upstream computation; use Python scalars, not 0-d mx.arrays, to avoid fp16→fp32
  promotion. `mx.async_eval` pipelines graph build with GPU execution.
  https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
- Eval discipline: evaluate once per natural boundary (per batch of texts / per query
  batch), never inside inner loops. https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50

## 5. Fused attention for small-model prefill

- `mx.fast.scaled_dot_product_attention(q, k, v, scale=, mask=)` is a fused
  flash-style kernel: softmax accumulated in fp32 regardless of input dtype; supports
  MHA/GQA/MQA (do NOT pre-tile k/v for GQA); mask = None | "causal" | boolean/additive
  array broadcastable to [B, N, T_q, T_kv]. For BERT use mask=None or an additive padding
  mask; scale = 1/sqrt(head_dim).
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- Behavior by regime: MLX routes decode (T_q=1) and prefill through different internal
  kernels; historically the most hand-tuned path was decode, with prefill/masked cases
  falling back to a generic (still fused) implementation — community kernels (cider,
  mlx-metal-kernels) exist precisely because prefill/GQA-decode had headroom (reported
  1.2–1.9x prefill, up to 1.6x decode SDPA on M5-class HW).
  https://github.com/Mininglamp-AI/cider
  https://github.com/manishklach/mlx-metal-kernels
  https://github.com/ml-explore/mlx/issues/2955 (FlashAttention/PagedAttention proposal thread)
- For BERT-length sequences (≤512 tokens, 12 heads, d=64) the fused SDPA is comfortably
  the right call vs composed softmax(QK^T)V — one dispatch, no T×T fp32 score
  materialization; short sequences make attention a small fraction of encoder time anyway
  (GEMMs in FFN dominate). Don't hand-roll attention here.
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- llama.cpp comparison: its Metal backend implements its own MSL flash-attention kernels,
  specialized at runtime via MTLFunctionConstantValues per head-dim/GQA-ratio, with
  in-kernel dequant of quantized KV and fp16/fp32 accumulator selection; separate kernel
  variants for large-batch (prefill) vs vector (decode) workloads. Notable datapoint:
  varying ubatch 512→2048 showed no prefill throughput change on M4 Max (compute-bound,
  headroom in power draw), so don't over-tune batch knobs.
  https://deepwiki.com/ggml-org/llama.cpp/8.2-flash-attention-and-optimizations
  https://github.com/ggml-org/llama.cpp/issues/22745
- Also use `mx.fast.layer_norm` (and rms_norm/rope where applicable) — fused, internally
  higher-precision accumulation, no manual upcasting needed.
  https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50

## Checklist for cpp/oraclemlx

1. fp16 everywhere; Python/C++ scalars (not 0-d arrays) to avoid fp32 promotion.
2. Encoder: mx.fast SDPA + layer_norm; bucket seq lengths; mx.compile per bucket.
3. Corpus: pre-normalized (N,512) fp16; `q @ C.T` → `argpartition` → take, one eval per query batch.
4. Only read back k indices/scores; never round-trip full score vectors to host.
5. Quantize encoder to 8-bit gs=64 only if profiling shows weight-bandwidth bound; regression-test embedding cosine drift first.
6. Custom Metal kernel only if dispatch overhead of score+select shows up in traces; build once, reuse.

---

## What we actually shipped (2026-07-08, M4 Max, branch mlx-fastpath)

Measured end-to-end on the warm daemon (HTTP, 200 mixed probes from
`eval/probes_1k.tsv`, fresh copies of the live ~625MB DB, 33.4k live +
9.6k dead + 19.6k paraphrase vectors @ 512d):

| metric (server-side) | before | after |
|---|---|---|
| p50 | 1996 ms | **26.6 ms** (75x) |
| p95 | 2364 ms | **48.3 ms** (49x) |
| mean / max | 1955 / 2656 ms | 27.2 / 160 ms |
| probes_1k hit@5 | 143/200 | 143/200 |
| top-5 result parity | — | 195/200 identical, mean jaccard 0.997 |
| selfeval era / continuity (n=100) | 98 / 100 | 98 / 100 |
| probes.tsv hit@5 (fresh DB) | 15/15 | 15/15 |

Where the time actually went (profiled, not guessed):

1. **~1.6s/query: missing index.** search() hydration counts live
   contradictions with a correlated subquery on `edges.dst` — full table scan
   per hydrated fact under modernc. Fix: `idx_edges_dst_type` (internal/store/db.go).
2. **~340ms/query: the three Go cosine scans** (SQL row scan + blob decode +
   float64 dots). Fix: persistent GPU vector store in cpp/oraclemlx
   (`omlx_vecs_load/topk/score`), fp16 [n,512] sets resident on Metal,
   one matmul + argpartition + take per arm; 0.51 ms/op for 33k x 512 top-60
   (BenchmarkMLXVecsTopK33k). as-of queries GPU-score the full corpus and
   filter validity-at-T host-side — exact, no approximation.
3. **~35ms/query: modernc bm25.** C SQLite runs the same FTS query 2.4x
   faster. Fix: `-tags sqlite_fts5` builds the read pool on mattn/go-sqlite3
   (C SQLite); modernc stays the writer + the default (no tag) build.
   The writer pool stays at 1 conn; reads go through a query_only pool
   (WAL: 1 writer + N readers), which also let the four probe arms
   (embed, live FTS, dead FTS, para FTS) actually run in parallel.
4. **~38ms/query: vec-store freshness signature** (COUNTs over blob-heavy
   tables) — now checked at most every 3s (sigTTL).

### Kernel / quantization decisions

- **No custom Metal kernel.** After the above, the embed forward (fp16 under
  mx::compile, mx::fast SDPA/layer_norm) is ~8-11 ms single-text and the FTS
  arm is the critical path; a fused embed+score kernel would shave nothing
  measurable. Matches the research above: for 50k x 512 the GEMM+argpartition
  is trivial and dispatch/eval dominates — solved by batching the three arms
  into single evals, not by hand-rolled kernels.
- **Quantization NOT adopted** (gate: parity cosine >= 0.995 AND faster).
  Measured on 500 real fact statements vs fp16:
  - 8-bit (affine, gs=64): parity mean 0.99982 / min 0.99973 — but SLOWER
    (13.4ms vs 8.1ms single; 1.90s vs 1.81s for 500 batched). At batch=1 on a
    110M encoder the fp16 GEMM is already bandwidth-trivial; quantized_matmul
    adds dequant overhead.
  - 4-bit: parity mean 0.950 / min 0.927 — fails the gate outright, consistent
    with the BERT-encoder sensitivity literature cited above.
  - `OMLX_QUANT_BITS=4|8` remains as an experiment knob (default off, fp16).
