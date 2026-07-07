// oraclemlx.cpp — full bert-base forward in MLX ops on Apple Metal.
//
// fp16 weights/compute; fp32 where accuracy is load-bearing: layer_norm and
// softmax accumulate in fp32 inside MLX fast:: kernels, and both heads
// (pooler/classifier softmax, mean-pool/proj/L2-norm) run in fp32.
// The whole forward is wrapped in mx::compile for kernel fusion; compile
// caches per input shape, so repeated single-item and batch calls are cheap.
//
// Weights come from mlx_weights.safetensors (scripts/export_safetensors.py):
// every 2D linear weight is (in, out) — plain x @ W + b, no transposes.

#include "oraclemlx.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {

thread_local std::string g_err;

constexpr int kLayers = 12;
constexpr int kHeads = 12;
constexpr int kDim = 768;

// OMLX_QUANT_BITS=4|8 quantizes every transformer linear (qkv, attn out, ffn)
// with mlx affine quantization (group 64). Unset/0 = fp16 (default). Embedding
// tables, layer norms, and the fp32 heads are never quantized.
int quant_bits() {
  const char* e = std::getenv("OMLX_QUANT_BITS");
  if (e == nullptr || *e == '\0') return 0;
  int b = std::atoi(e);
  if (b != 4 && b != 8) {
    throw std::runtime_error("OMLX_QUANT_BITS must be 4 or 8 (got '" +
                             std::string(e) + "')");
  }
  return b;
}

// Lin: an (in, out) linear, stored fp16 or affine-quantized. apply() is
// x @ W + b either way (quantized_matmul takes the (out, in) transpose).
struct Lin {
  mx::array w, b, scales, qbiases;
  int bits = 0;
  Lin() : w(mx::array({0.0f})), b(w), scales(w), qbiases(w) {}
  Lin(mx::array w_, mx::array b_, int bits_) : Lin() {
    b = std::move(b_);
    bits = bits_;
    if (bits == 0) {
      w = std::move(w_);
      return;
    }
    auto q = mx::quantize(mx::transpose(w_), 64, bits);
    w = q[0];
    scales = q[1];
    qbiases = q[2];
  }
  mx::array apply(const mx::array& x) const {
    if (bits == 0) return mx::addmm(b, x, w);
    return mx::quantized_matmul(x, w, scales, qbiases, /*transpose=*/true, 64,
                                bits) +
           b;
  }
};

struct Bert {
  // embeddings (fp16 except where noted)
  mx::array emb_word, emb_pos, emb_type, emb_ln_w, emb_ln_b;
  // per layer: fused qkv, attn out, ffn (fp16 or quantized), layer norms
  std::vector<Lin> qkv, attn_out, ffn_in, ffn_out;
  std::vector<mx::array> ln1_w, ln1_b, ln2_w, ln2_b;
  // heads (fp32)
  bool is_judge = false;
  mx::array pooler_w, pooler_b, cls_w, cls_b;  // judge
  mx::array proj;                              // embedder
  std::function<std::vector<mx::array>(const std::vector<mx::array>&)> fwd;

  Bert()
      : emb_word(mx::array({0.0f})), emb_pos(emb_word), emb_type(emb_word),
        emb_ln_w(emb_word), emb_ln_b(emb_word), pooler_w(emb_word),
        pooler_b(emb_word), cls_w(emb_word), cls_b(emb_word), proj(emb_word) {}
};

Bert g_judge, g_embed;
bool g_ready = false;
std::mutex g_mu;

mx::array want(std::unordered_map<std::string, mx::array>& w,
               const std::string& k, mx::Dtype dt) {
  auto it = w.find(k);
  if (it == w.end()) {
    throw std::runtime_error("missing tensor " + k + " in safetensors");
  }
  return mx::astype(it->second, dt);
}

void load_bert(Bert& b, const std::string& path, bool judge) {
  auto [w, meta] = mx::load_safetensors(path);
  const auto f16 = mx::float16;
  const int qb = quant_bits();
  b.is_judge = judge;
  b.emb_word = want(w, "emb.word", f16);
  b.emb_pos = want(w, "emb.pos", f16);
  b.emb_type = want(w, "emb.type", f16);
  b.emb_ln_w = want(w, "emb.ln.w", f16);
  b.emb_ln_b = want(w, "emb.ln.b", f16);
  for (int i = 0; i < kLayers; i++) {
    auto L = "L" + std::to_string(i) + ".";
    // fused qkv: concat (in,out) weights along out axis -> [768, 2304]
    b.qkv.emplace_back(
        mx::concatenate({want(w, L + "q.w", f16), want(w, L + "k.w", f16),
                         want(w, L + "v.w", f16)},
                        1),
        mx::concatenate({want(w, L + "q.b", f16), want(w, L + "k.b", f16),
                         want(w, L + "v.b", f16)}),
        qb);
    b.attn_out.emplace_back(want(w, L + "attn_out.w", f16),
                            want(w, L + "attn_out.b", f16), qb);
    b.ln1_w.push_back(want(w, L + "attn_ln.w", f16));
    b.ln1_b.push_back(want(w, L + "attn_ln.b", f16));
    b.ffn_in.emplace_back(want(w, L + "ffn_in.w", f16),
                          want(w, L + "ffn_in.b", f16), qb);
    b.ffn_out.emplace_back(want(w, L + "ffn_out.w", f16),
                           want(w, L + "ffn_out.b", f16), qb);
    b.ln2_w.push_back(want(w, L + "out_ln.w", f16));
    b.ln2_b.push_back(want(w, L + "out_ln.b", f16));
  }
  if (judge) {
    b.pooler_w = want(w, "pooler.w", mx::float32);
    b.pooler_b = want(w, "pooler.b", mx::float32);
    b.cls_w = want(w, "cls.w", mx::float32);
    b.cls_b = want(w, "cls.b", mx::float32);
  } else {
    b.proj = want(w, "proj", mx::float32);
  }

  // inputs: ids [B,L] i32, mask [B,L] i32, types [B,L] i32
  auto forward = [&b](const std::vector<mx::array>& in) -> std::vector<mx::array> {
    const auto& ids = in[0];
    const auto& mask = in[1];
    const auto& types = in[2];
    int L = ids.shape(1);

    auto x = mx::take(b.emb_word, ids, 0) + mx::take(b.emb_type, types, 0) +
             mx::slice(b.emb_pos, {0, 0}, {L, kDim});
    x = mx::fast::layer_norm(x, b.emb_ln_w, b.emb_ln_b, 1e-12f);

    // additive attention mask [B,1,1,L]; fp16-safe -30000 for pad positions
    auto amask =
        mx::reshape((mx::astype(mask, mx::float16) - mx::array(1.0f, mx::float16)) *
                        mx::array(30000.0f, mx::float16),
                    {-1, 1, 1, L});

    const float scale = 1.0f / std::sqrt(static_cast<float>(kDim / kHeads));
    for (int i = 0; i < kLayers; i++) {
      auto qkv = b.qkv[i].apply(x);
      auto parts = mx::split(qkv, 3, 2);
      auto heads = [L](mx::array a) {
        return mx::transpose(mx::reshape(a, {-1, L, kHeads, kDim / kHeads}),
                             {0, 2, 1, 3});
      };
      auto o = mx::fast::scaled_dot_product_attention(
          heads(parts[0]), heads(parts[1]), heads(parts[2]), scale, "array",
          {amask});
      o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {-1, L, kDim});
      x = mx::fast::layer_norm(x + b.attn_out[i].apply(o), b.ln1_w[i],
                               b.ln1_b[i], 1e-12f);
      auto h = b.ffn_in[i].apply(x);
      h = h * mx::array(0.5f, mx::float16) *
          (mx::array(1.0f, mx::float16) +
           mx::erf(h * mx::array(0.7071067811865476f, mx::float16)));
      x = mx::fast::layer_norm(x + b.ffn_out[i].apply(h), b.ln2_w[i],
                               b.ln2_b[i], 1e-12f);
    }

    if (b.is_judge) {
      // [CLS] -> pooler tanh -> classifier -> softmax, all fp32
      auto cls = mx::astype(mx::squeeze(mx::slice(x, {0, 0, 0},
                                                  {x.shape(0), 1, kDim}),
                                        1),
                            mx::float32);
      auto pooled = mx::tanh(mx::addmm(b.pooler_b, cls, b.pooler_w));
      auto logits = mx::addmm(b.cls_b, pooled, b.cls_w);
      return {mx::softmax(logits, -1, /*precise=*/true)};
    }
    // masked mean-pool -> 512 proj -> L2 normalize, fp32
    auto xf = mx::astype(x, mx::float32);
    auto mf = mx::reshape(mx::astype(mask, mx::float32), {-1, L, 1});
    auto pooled = mx::sum(xf * mf, 1) /
                  mx::maximum(mx::sum(mf, 1), mx::array(1.0f));
    auto e = mx::matmul(pooled, b.proj);
    return {e / mx::linalg::norm(e, 2.0, std::vector<int>{-1},
                                 /*keepdims=*/true)};
  };
  b.fwd = mx::compile(forward);
}

std::vector<mx::array> pack(const int32_t* ids, const int32_t* mask,
                            const int32_t* types, int n, int seq) {
  mx::Shape shp{n, seq};
  auto a = [&](const int32_t* p) {
    return mx::array(p, shp, mx::int32);
  };
  auto t = types != nullptr
               ? a(types)
               : mx::zeros(shp, mx::int32);
  return {a(ids), a(mask), t};
}

int run(Bert& b, const int32_t* ids, const int32_t* mask, const int32_t* types,
        int n, int seq, int out_dim, float* out) {
  if (!g_ready) {
    g_err = "oraclemlx: omlx_init not called (or failed)";
    return 1;
  }
  if (n <= 0 || ids == nullptr || mask == nullptr || out == nullptr) {
    g_err = "oraclemlx: bad arguments";
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lk(g_mu);
    auto r = b.fwd(pack(ids, mask, types, n, seq))[0];
    r = mx::astype(mx::contiguous(r), mx::float32);
    mx::eval(r);
    if (r.shape(0) != n || r.shape(1) != out_dim) {
      throw std::runtime_error("unexpected output shape");
    }
    std::memcpy(out, r.data<float>(), sizeof(float) * n * out_dim);
    return 0;
  } catch (const std::exception& e) {
    g_err = std::string("oraclemlx: ") + e.what();
    return 1;
  }
}

// ---- persistent vector store: fp16 [n,dim] sets scored by one GPU pass ----

struct VecSet {
  mx::array vecs;  // fp16 [n, dim]
  int n = 0, dim = 0;
  VecSet() : vecs(mx::array({0.0f})) {}
};

std::unordered_map<std::string, VecSet> g_vecsets;
std::mutex g_vecs_mu;

}  // namespace

extern "C" {

int omlx_vecs_load(const char* name, const float* data, int n, int dim) {
  if (name == nullptr || (n > 0 && (data == nullptr || dim <= 0)) || n < 0) {
    g_err = "oraclemlx: vecs_load bad arguments";
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lk(g_vecs_mu);
    mx::set_default_device(mx::Device::gpu);
    if (n == 0) {
      g_vecsets.erase(name);
      return 0;
    }
    VecSet s;
    s.n = n;
    s.dim = dim;
    // copy host fp32 -> device fp16 once; eval so the upload cost is paid here
    s.vecs = mx::astype(mx::array(data, mx::Shape{n, dim}, mx::float32),
                        mx::float16);
    mx::eval(s.vecs);
    g_vecsets.insert_or_assign(name, std::move(s));
    return 0;
  } catch (const std::exception& e) {
    g_err = std::string("oraclemlx: vecs_load: ") + e.what();
    return 1;
  }
}

int omlx_vecs_score(const char* name, const float* query, int dim,
                    float* out_scores, int* out_n) {
  if (name == nullptr || query == nullptr || out_scores == nullptr ||
      out_n == nullptr) {
    g_err = "oraclemlx: vecs_score bad arguments";
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lk(g_vecs_mu);
    auto it = g_vecsets.find(name);
    if (it == g_vecsets.end()) {
      g_err = std::string("oraclemlx: vecs_score: set '") + name +
              "' not loaded";
      return 1;
    }
    VecSet& s = it->second;
    if (dim != s.dim) {
      g_err = "oraclemlx: vecs_score: dim mismatch";
      return 1;
    }
    if (s.n > *out_n) {
      g_err = "oraclemlx: vecs_score: out_scores capacity too small";
      return 1;
    }
    auto q = mx::astype(mx::array(query, mx::Shape{dim, 1}, mx::float32),
                        mx::float16);
    auto scores = mx::squeeze(
        mx::astype(mx::matmul(s.vecs, q), mx::float32), 1);
    scores = mx::contiguous(scores);
    mx::eval(scores);
    std::memcpy(out_scores, scores.data<float>(), sizeof(float) * s.n);
    *out_n = s.n;
    return 0;
  } catch (const std::exception& e) {
    g_err = std::string("oraclemlx: vecs_score: ") + e.what();
    return 1;
  }
}

int omlx_vecs_topk(const char* name, const float* query, int dim, int k,
                   int32_t* out_idx, float* out_scores, int* out_n) {
  if (name == nullptr || query == nullptr || out_idx == nullptr ||
      out_scores == nullptr || out_n == nullptr || k <= 0) {
    g_err = "oraclemlx: vecs_topk bad arguments";
    return 1;
  }
  try {
    std::lock_guard<std::mutex> lk(g_vecs_mu);
    auto it = g_vecsets.find(name);
    if (it == g_vecsets.end()) {
      g_err = std::string("oraclemlx: vecs_topk: set '") + name +
              "' not loaded";
      return 1;
    }
    VecSet& s = it->second;
    if (dim != s.dim) {
      g_err = "oraclemlx: vecs_topk: dim mismatch";
      return 1;
    }
    if (k > s.n) k = s.n;
    auto q = mx::astype(mx::array(query, mx::Shape{dim, 1}, mx::float32),
                        mx::float16);
    // scores fp32 [n]; fp16 matmul, fp32 select
    auto scores = mx::squeeze(
        mx::astype(mx::matmul(s.vecs, q), mx::float32), 1);
    mx::array idx = mx::array({0});
    if (k == s.n) {
      idx = mx::argsort(mx::negative(scores), 0);
    } else {
      // partition the top k to the front, then order just those k
      auto part = mx::slice(mx::argpartition(mx::negative(scores), k - 1, 0),
                            {0}, {k});
      auto vals = mx::take(scores, part, 0);
      idx = mx::take(part, mx::argsort(mx::negative(vals), 0), 0);
    }
    idx = mx::astype(idx, mx::int32);
    auto top = mx::take(scores, idx, 0);
    mx::eval(idx, top);
    std::memcpy(out_idx, idx.data<int32_t>(), sizeof(int32_t) * k);
    std::memcpy(out_scores, top.data<float>(), sizeof(float) * k);
    *out_n = k;
    return 0;
  } catch (const std::exception& e) {
    g_err = std::string("oraclemlx: vecs_topk: ") + e.what();
    return 1;
  }
}

int omlx_init(const char* models_dir) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_ready) return 0;
  if (models_dir == nullptr) {
    g_err = "oraclemlx: models_dir is null";
    return 1;
  }
  try {
    mx::set_default_device(mx::Device::gpu);
    std::string root(models_dir);
    load_bert(g_judge, root + "/judge_v2_onnx/mlx_weights.safetensors", true);
    load_bert(g_embed, root + "/embedder_v3_onnx/mlx_weights.safetensors",
              false);
    g_ready = true;
    return 0;
  } catch (const std::exception& e) {
    g_err = std::string("oraclemlx: init: ") + e.what();
    return 1;
  }
}

int omlx_embed(const int32_t* ids, const int32_t* mask, int n, float* out) {
  return run(g_embed, ids, mask, nullptr, n, OMLX_EMBED_SEQ, OMLX_EMBED_DIM,
             out);
}

int omlx_judge(const int32_t* ids, const int32_t* mask,
               const int32_t* type_ids, int n, float* out) {
  if (type_ids == nullptr) {
    g_err = "oraclemlx: judge requires token_type_ids";
    return 1;
  }
  return run(g_judge, ids, mask, type_ids, n, OMLX_JUDGE_SEQ,
             OMLX_JUDGE_CLASSES, out);
}

const char* omlx_last_error(void) { return g_err.c_str(); }

}  // extern "C"
