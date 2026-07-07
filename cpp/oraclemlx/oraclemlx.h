/* oraclemlx.h — C API for oracle's MLX (Apple Metal) bert inference engine.
 *
 * Models: judge_v2 (bert-base 3-class classifier, seq 256) and embedder_v3
 * (e5-base + masked mean-pool + 768x512 projection + L2 norm, seq 128).
 * Token ids come from the caller (Go WordPiece tokenizer) — no tokenization here.
 *
 * All functions return 0 on success, nonzero on failure; omlx_last_error()
 * returns a human-readable message for the last failure on the calling thread.
 */
#ifndef ORACLEMLX_H
#define ORACLEMLX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMLX_EMBED_SEQ 128
#define OMLX_EMBED_DIM 512
#define OMLX_JUDGE_SEQ 256
#define OMLX_JUDGE_CLASSES 3

/* Load both models from models_dir (expects
 * <models_dir>/{judge_v2_onnx,embedder_v3_onnx}/mlx_weights.safetensors,
 * produced by scripts/export_safetensors.py). Idempotent. */
int omlx_init(const char* models_dir);

/* Embed n texts. ids/mask are packed row-major [n][OMLX_EMBED_SEQ] int32.
 * out receives n*OMLX_EMBED_DIM floats (L2-normalized rows). */
int omlx_embed(const int32_t* ids, const int32_t* mask, int n, float* out);

/* Judge n pairs. ids/mask/type_ids are packed [n][OMLX_JUDGE_SEQ] int32.
 * out receives n*OMLX_JUDGE_CLASSES softmax probabilities. */
int omlx_judge(const int32_t* ids, const int32_t* mask, const int32_t* type_ids,
               int n, float* out);

/* ---- persistent vector store (brute-force cosine top-k on GPU) ----
 *
 * omlx_vecs_load uploads n row-vectors of `dim` floats under `name` (kept fp16
 * on the GPU; rows are assumed L2-normalized so dot == cosine). Reloading a
 * name replaces the set. n == 0 clears the set. Usable before/without
 * omlx_init.
 *
 * omlx_vecs_topk scores `query` (dim floats) against the named set and writes
 * the top-k row indices (descending score) to out_idx and their fp32 scores to
 * out_scores. k is clamped to n; the effective k is written to *out_n. One
 * matmul + argpartition + sort, a single GPU eval. */
int omlx_vecs_load(const char* name, const float* data, int n, int dim);
int omlx_vecs_topk(const char* name, const float* query, int dim, int k,
                   int32_t* out_idx, float* out_scores, int* out_n);

/* omlx_vecs_score writes the full score vector (fp32 dots) for `query`
 * against the named set. *out_n carries the out_scores capacity in and the
 * set's row count out; a set larger than the capacity is an error (no
 * truncated writes). For callers that filter rows host-side before top-k. */
int omlx_vecs_score(const char* name, const float* query, int dim,
                    float* out_scores, int* out_n);

/* Message for the last error on this thread ("" if none). */
const char* omlx_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* ORACLEMLX_H */
