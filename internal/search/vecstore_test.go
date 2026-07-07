package search

// vecstore_test.go — GPU vector store parity vs the Go cosine reference, plus
// a micro-benchmark. Skips with a NOTICE when the MLX dylib is absent.

import (
	"math"
	"math/rand"
	"oracle/internal/infer"
	"oracle/internal/store"
	"sort"
	"testing"
)

// randUnitVecs builds n L2-normalized dim vectors (deterministic seed).
func randUnitVecs(n, dim int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		var norm float64
		for j := range v {
			v[j] = float32(r.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] = float32(float64(v[j]) / norm)
		}
		out[i] = v
	}
	return out
}

func TestMLXVecsTopKParity(t *testing.T) {
	infer.RequireMLX(t)
	const n, dim, k = 5000, embedDims, 60
	vecs := randUnitVecs(n, dim, 1)
	flat := make([]float32, 0, n*dim)
	for _, v := range vecs {
		flat = append(flat, v...)
	}
	if err := infer.MLXVecsLoad("test_parity", flat, n, dim); err != nil {
		t.Fatalf("mlxVecsLoad: %v", err)
	}
	q := randUnitVecs(1, dim, 2)[0]

	// Go reference: full sort by float64 dot
	type pair struct {
		idx int
		sim float64
	}
	ref := make([]pair, n)
	for i, v := range vecs {
		ref[i] = pair{i, store.Dot(q, v)}
	}
	sort.Slice(ref, func(i, j int) bool { return ref[i].sim > ref[j].sim })

	idx, scores, err := infer.MLXVecsTopK("test_parity", q, k)
	if err != nil {
		t.Fatalf("mlxVecsTopK: %v", err)
	}
	if len(idx) != k || len(scores) != k {
		t.Fatalf("got %d idx / %d scores, want %d", len(idx), len(scores), k)
	}
	// fp16 storage: allow tiny score error and rank swaps between near-ties.
	refSet := map[int]bool{}
	for _, p := range ref[:k+5] { // small tie slack at the boundary
		refSet[p.idx] = true
	}
	miss := 0
	for i, j := range idx {
		if !refSet[int(j)] {
			miss++
		}
		if d := math.Abs(float64(scores[i]) - ref[i].sim); d > 5e-3 {
			t.Fatalf("rank %d: score %f vs ref %f (|d|=%g > 5e-3)", i, scores[i], ref[i].sim, d)
		}
		if i > 0 && scores[i] > scores[i-1] {
			t.Fatalf("scores not descending at %d", i)
		}
	}
	if miss > 2 {
		t.Fatalf("%d of top-%d GPU hits not in reference top-%d", miss, k, k+5)
	}
}

func TestMLXVecsLoadClearAndErrors(t *testing.T) {
	infer.RequireMLX(t)
	if err := infer.MLXVecsLoad("test_clear", make([]float32, 4*embedDims), 4, embedDims); err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := infer.MLXVecsLoad("test_clear", nil, 0, embedDims); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, _, err := infer.MLXVecsTopK("test_clear", make([]float32, embedDims), 5); err == nil {
		t.Fatal("topk on a cleared set must fail loudly")
	}
	if _, _, err := infer.MLXVecsTopK("never_loaded", make([]float32, embedDims), 5); err == nil {
		t.Fatal("topk on an unknown set must fail loudly")
	}
	if err := infer.MLXVecsLoad("test_bad", make([]float32, 7), 4, embedDims); err == nil {
		t.Fatal("size-mismatched load must fail loudly")
	}
}

func BenchmarkMLXVecsTopK33k(b *testing.B) {
	infer.RequireMLX(b)
	const n, dim = 33000, embedDims
	flat := make([]float32, 0, n*dim)
	for _, v := range randUnitVecs(n, dim, 3) {
		flat = append(flat, v...)
	}
	if err := infer.MLXVecsLoad("bench_33k", flat, n, dim); err != nil {
		b.Fatalf("load: %v", err)
	}
	q := randUnitVecs(1, dim, 4)[0]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := infer.MLXVecsTopK("bench_33k", q, 60); err != nil {
			b.Fatal(err)
		}
	}
}

func TestMLXVecsScoreParity(t *testing.T) {
	infer.RequireMLX(t)
	const n, dim = 3000, embedDims
	vecs := randUnitVecs(n, dim, 7)
	flat := make([]float32, 0, n*dim)
	for _, v := range vecs {
		flat = append(flat, v...)
	}
	if err := infer.MLXVecsLoad("test_score", flat, n, dim); err != nil {
		t.Fatalf("load: %v", err)
	}
	q := randUnitVecs(1, dim, 8)[0]
	scores, err := infer.MLXVecsScore("test_score", q, n)
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	for i, v := range vecs {
		if d := math.Abs(float64(scores[i]) - store.Dot(q, v)); d > 5e-3 {
			t.Fatalf("row %d: |gpu-ref| = %g > 5e-3", i, d)
		}
	}
	if _, err := infer.MLXVecsScore("test_score", q, n-1); err == nil {
		t.Fatal("undersized capacity must fail loudly")
	}
}
