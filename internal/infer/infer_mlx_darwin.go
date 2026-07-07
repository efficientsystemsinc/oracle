//go:build darwin && cgo

package infer

// infer_mlx_darwin.go — cgo binding to the MLX (Apple Metal) inference engine
// in cpp/oraclemlx (liboraclemlx.dylib). Loaded via dlopen at runtime so the
// Go binary builds and tests pass on machines without the dylib.
//
// Dylib discovery: ORACLE_MLX_DYLIB, then ~/.oracle/models/lib/, then the
// in-repo cmake build dir. Missing dylib with ORACLE_MLX=1 is a loud error
// (ADR-004: no silent fallback to ORT).

/*
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>

typedef int (*omlx_init_t)(const char*);
typedef int (*omlx_embed_t)(const int32_t*, const int32_t*, int, float*);
typedef int (*omlx_judge_t)(const int32_t*, const int32_t*, const int32_t*, int, float*);
typedef const char* (*omlx_err_t)(void);
typedef int (*omlx_vecs_load_t)(const char*, const float*, int, int);
typedef int (*omlx_vecs_topk_t)(const char*, const float*, int, int, int32_t*, float*, int*);
typedef int (*omlx_vecs_score_t)(const char*, const float*, int, float*, int*);

static void* omlx_handle;
static omlx_init_t p_init;
static omlx_embed_t p_embed;
static omlx_judge_t p_judge;
static omlx_err_t p_err;
static omlx_vecs_load_t p_vecs_load;
static omlx_vecs_topk_t p_vecs_topk;
static omlx_vecs_score_t p_vecs_score;

static int omlx_open(const char* path) {
	omlx_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (omlx_handle == NULL) return 1;
	p_init  = (omlx_init_t)dlsym(omlx_handle, "omlx_init");
	p_embed = (omlx_embed_t)dlsym(omlx_handle, "omlx_embed");
	p_judge = (omlx_judge_t)dlsym(omlx_handle, "omlx_judge");
	p_err   = (omlx_err_t)dlsym(omlx_handle, "omlx_last_error");
	// vec-store symbols are newer than embed/judge; wrappers check for NULL so
	// an old dylib still serves embed/judge and fails loudly only on vec calls.
	p_vecs_load = (omlx_vecs_load_t)dlsym(omlx_handle, "omlx_vecs_load");
	p_vecs_topk = (omlx_vecs_topk_t)dlsym(omlx_handle, "omlx_vecs_topk");
	p_vecs_score = (omlx_vecs_score_t)dlsym(omlx_handle, "omlx_vecs_score");
	if (p_init == NULL || p_embed == NULL || p_judge == NULL || p_err == NULL) return 2;
	return 0;
}
static int omlx_has_vecs(void) {
	return p_vecs_load != NULL && p_vecs_topk != NULL && p_vecs_score != NULL;
}
static int c_omlx_vecs_score(const char* name, const float* q, int dim,
                             float* out_scores, int* out_n) {
	return p_vecs_score(name, q, dim, out_scores, out_n);
}
static int c_omlx_vecs_load(const char* name, const float* data, int n, int dim) {
	return p_vecs_load(name, data, n, dim);
}
static int c_omlx_vecs_topk(const char* name, const float* q, int dim, int k,
                            int32_t* out_idx, float* out_scores, int* out_n) {
	return p_vecs_topk(name, q, dim, k, out_idx, out_scores, out_n);
}
static int c_omlx_init(const char* dir) { return p_init(dir); }
static int c_omlx_embed(const int32_t* ids, const int32_t* mask, int n, float* out) {
	return p_embed(ids, mask, n, out);
}
static int c_omlx_judge(const int32_t* ids, const int32_t* mask, const int32_t* tt, int n, float* out) {
	return p_judge(ids, mask, tt, n, out);
}
static const char* c_omlx_err(void) {
	if (p_err != NULL) return p_err();
	const char* d = dlerror();
	return d != NULL ? d : "unknown dlopen error";
}
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"
)

const (
	mlxEmbedSeq = 128 // OMLX_EMBED_SEQ (embedder ONNX graph is static at 128)
	mlxEmbedDim = 512
	mlxJudgeSeq = 256 // OMLX_JUDGE_SEQ
)

var (
	mlxOnce    sync.Once
	mlxInitErr error
	mlxEmbTok  *wordPieceTokenizer
	mlxJudTok  *wordPieceTokenizer
	mlxMu      sync.Mutex
)

func findMLXDylib() (string, error) {
	if p := os.Getenv("ORACLE_MLX_DYLIB"); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("infer_mlx: ORACLE_MLX_DYLIB=%s not readable: %w", p, err)
		}
		return p, nil
	}
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, ".oracle", "models", "lib", "liboraclemlx.dylib"),
		filepath.Join("cpp", "oraclemlx", "build", "liboraclemlx.dylib"),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("infer_mlx: liboraclemlx.dylib not found (tried %v); "+
		"build cpp/oraclemlx and install to ~/.oracle/models/lib/, or set ORACLE_MLX_DYLIB", candidates)
}

func mlxInit() error {
	mlxOnce.Do(func() {
		if err := ensureModels("lib", "judge_v2_onnx", "embedder_v3_onnx"); err != nil {
			mlxInitErr = err
			return
		}
		lib, err := findMLXDylib()
		if err != nil {
			mlxInitErr = err
			return
		}
		cLib := C.CString(lib)
		defer C.free(unsafe.Pointer(cLib))
		if rc := C.omlx_open(cLib); rc != 0 {
			mlxInitErr = fmt.Errorf("infer_mlx: dlopen %s failed (rc=%d): %s",
				lib, int(rc), C.GoString(C.c_omlx_err()))
			return
		}
		modelsDir := filepath.Dir(oracleModelDir("x"))
		cDir := C.CString(modelsDir)
		defer C.free(unsafe.Pointer(cDir))
		if rc := C.c_omlx_init(cDir); rc != 0 {
			mlxInitErr = fmt.Errorf("infer_mlx: omlx_init(%s): %s", modelsDir, C.GoString(C.c_omlx_err()))
			return
		}
		mlxEmbTok, err = loadWordPieceTokenizer(filepath.Join(oracleModelDir("embedder_v3_onnx"), "vocab.txt"))
		if err != nil {
			mlxInitErr = err
			return
		}
		mlxJudTok, err = loadWordPieceTokenizer(filepath.Join(oracleModelDir("judge_v2_onnx"), "vocab.txt"))
		if err != nil {
			mlxInitErr = err
		}
	})
	return mlxInitErr
}

func mlxAvailable() (bool, string) {
	if err := mlxInit(); err != nil {
		return false, err.Error()
	}
	return true, ""
}

func toInt32(xs []int64, dst []int32) []int32 {
	for _, x := range xs {
		dst = append(dst, int32(x))
	}
	return dst
}

// embedMLX embeds texts on Metal; rows are L2-normalized 512-d vectors.
func embedMLX(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("infer_mlx: embedMLX called with no texts")
	}
	if err := mlxInit(); err != nil {
		return nil, err
	}
	n := len(texts)
	ids := make([]int32, 0, n*mlxEmbedSeq)
	mask := make([]int32, 0, n*mlxEmbedSeq)
	for _, t := range texts {
		ii, am, _ := mlxEmbTok.Encode(t, "", mlxEmbedSeq)
		ids = toInt32(ii, ids)
		mask = toInt32(am, mask)
	}
	out := make([]float32, n*mlxEmbedDim)
	mlxMu.Lock()
	rc := C.c_omlx_embed((*C.int32_t)(unsafe.Pointer(&ids[0])),
		(*C.int32_t)(unsafe.Pointer(&mask[0])), C.int(n),
		(*C.float)(unsafe.Pointer(&out[0])))
	errMsg := C.GoString(C.c_omlx_err())
	mlxMu.Unlock()
	if rc != 0 {
		return nil, fmt.Errorf("infer_mlx: omlx_embed: %s", errMsg)
	}
	res := make([][]float32, n)
	for i := range res {
		res[i] = out[i*mlxEmbedDim : (i+1)*mlxEmbedDim : (i+1)*mlxEmbedDim]
	}
	return res, nil
}

// judgeMLXBatch judges (a[i], b[i]) pairs; returns softmax probs per pair.
func judgeMLXBatch(a, b []string) ([][3]float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return nil, fmt.Errorf("infer_mlx: judgeMLXBatch bad batch (%d, %d)", len(a), len(b))
	}
	if err := mlxInit(); err != nil {
		return nil, err
	}
	n := len(a)
	ids := make([]int32, 0, n*mlxJudgeSeq)
	mask := make([]int32, 0, n*mlxJudgeSeq)
	tts := make([]int32, 0, n*mlxJudgeSeq)
	for i := range a {
		ii, am, tt := mlxJudTok.Encode(a[i], b[i], mlxJudgeSeq)
		ids = toInt32(ii, ids)
		mask = toInt32(am, mask)
		tts = toInt32(tt, tts)
	}
	out := make([]float32, n*3)
	mlxMu.Lock()
	rc := C.c_omlx_judge((*C.int32_t)(unsafe.Pointer(&ids[0])),
		(*C.int32_t)(unsafe.Pointer(&mask[0])),
		(*C.int32_t)(unsafe.Pointer(&tts[0])), C.int(n),
		(*C.float)(unsafe.Pointer(&out[0])))
	errMsg := C.GoString(C.c_omlx_err())
	mlxMu.Unlock()
	if rc != 0 {
		return nil, fmt.Errorf("infer_mlx: omlx_judge: %s", errMsg)
	}
	res := make([][3]float64, n)
	for i := range res {
		for j := 0; j < 3; j++ {
			res[i][j] = float64(out[i*3+j])
		}
	}
	return res, nil
}

// MLXVecsLoad uploads a named set of n dim-length L2-normalized row vectors
// (flat, row-major) to the GPU vector store. n==0 clears the set.
func MLXVecsLoad(name string, flat []float32, n, dim int) error {
	if err := mlxInit(); err != nil {
		return err
	}
	if C.omlx_has_vecs() == 0 {
		return fmt.Errorf("infer_mlx: dylib has no vector-store symbols — rebuild cpp/oraclemlx")
	}
	if n*dim != len(flat) {
		return fmt.Errorf("infer_mlx: MLXVecsLoad %s: %d*%d != %d floats", name, n, dim, len(flat))
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var p *C.float
	if n > 0 {
		p = (*C.float)(unsafe.Pointer(&flat[0]))
	}
	mlxMu.Lock()
	rc := C.c_omlx_vecs_load(cName, p, C.int(n), C.int(dim))
	errMsg := C.GoString(C.c_omlx_err())
	mlxMu.Unlock()
	if rc != 0 {
		return fmt.Errorf("infer_mlx: omlx_vecs_load(%s): %s", name, errMsg)
	}
	return nil
}

// MLXVecsTopK returns the top-k row indices (descending cosine) and scores of
// the query vector against a loaded set. k is clamped to the set size.
func MLXVecsTopK(name string, q []float32, k int) ([]int32, []float32, error) {
	if err := mlxInit(); err != nil {
		return nil, nil, err
	}
	if C.omlx_has_vecs() == 0 {
		return nil, nil, fmt.Errorf("infer_mlx: dylib has no vector-store symbols — rebuild cpp/oraclemlx")
	}
	if len(q) == 0 || k <= 0 {
		return nil, nil, fmt.Errorf("infer_mlx: MLXVecsTopK(%s): empty query or k<=0", name)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	idx := make([]int32, k)
	scores := make([]float32, k)
	var got C.int
	mlxMu.Lock()
	rc := C.c_omlx_vecs_topk(cName, (*C.float)(unsafe.Pointer(&q[0])),
		C.int(len(q)), C.int(k),
		(*C.int32_t)(unsafe.Pointer(&idx[0])),
		(*C.float)(unsafe.Pointer(&scores[0])), &got)
	errMsg := C.GoString(C.c_omlx_err())
	mlxMu.Unlock()
	if rc != 0 {
		return nil, nil, fmt.Errorf("infer_mlx: omlx_vecs_topk(%s): %s", name, errMsg)
	}
	return idx[:got], scores[:got], nil
}

// MLXVecsScore returns the full fp32 score vector of q against a loaded set
// (row order = load order). n must be the set's row count.
func MLXVecsScore(name string, q []float32, n int) ([]float32, error) {
	if err := mlxInit(); err != nil {
		return nil, err
	}
	if C.omlx_has_vecs() == 0 {
		return nil, fmt.Errorf("infer_mlx: dylib has no vector-store symbols — rebuild cpp/oraclemlx")
	}
	if len(q) == 0 || n <= 0 {
		return nil, fmt.Errorf("infer_mlx: MLXVecsScore(%s): empty query or n<=0", name)
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	scores := make([]float32, n)
	got := C.int(n) // in: capacity; out: set row count
	mlxMu.Lock()
	rc := C.c_omlx_vecs_score(cName, (*C.float)(unsafe.Pointer(&q[0])),
		C.int(len(q)), (*C.float)(unsafe.Pointer(&scores[0])), &got)
	errMsg := C.GoString(C.c_omlx_err())
	mlxMu.Unlock()
	if rc != 0 {
		return nil, fmt.Errorf("infer_mlx: omlx_vecs_score(%s): %s", name, errMsg)
	}
	if int(got) != n {
		return nil, fmt.Errorf("infer_mlx: omlx_vecs_score(%s): set has %d rows, caller expected %d", name, int(got), n)
	}
	return scores, nil
}

// judgeMLX judges one (oldSide, newSide) pair.
func judgeMLX(oldSide, newSide string) ([3]float64, error) {
	r, err := judgeMLXBatch([]string{oldSide}, []string{newSide})
	if err != nil {
		return [3]float64{}, err
	}
	return r[0], nil
}
