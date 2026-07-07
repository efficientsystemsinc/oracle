package search

// veccpu.go — portable in-RAM vector engine + the engine seam for vecstore.
//
// vecstore.go needs three primitives: load a named fp set, top-k by dot, and
// full score. On darwin+cgo with ORACLE_MLX=1 those run on Metal
// (infer_mlx_darwin.go); everywhere else this file serves them from plain Go:
// flat float32 rows in RAM, goroutine-parallel dot products, a size-k heap.
// Corpus rows are L2-normalized at write time (embed.go normalize), so dot IS
// cosine — scores here are float64-accumulated dots, identical to the SQL
// path's dot() and at least as exact as the fp16 Metal kernel.
//
// At oracle's scale this is not a compromise: 100k facts x 512 dims is ~51M
// multiply-adds, low single-digit ms across a few cores — the win over the
// SQL path (per-query table scan + blob decode) is the same one the GPU store
// bought on macOS.

import (
	"fmt"
	"oracle/internal/infer"
	"oracle/internal/store"
	"runtime"
	"sort"
	"sync"
)

type cpuVecSet struct {
	flat []float32
	n    int
	dim  int
}

var (
	cpuVecMu   sync.RWMutex
	cpuVecSets = map[string]cpuVecSet{}
)

// vecsLoad / vecsTopK / vecsScore — the engine seam vecstore.go calls.
// ORACLE_MLX=1 routes to Metal (loud error off-darwin, per ADR-004: the flag
// asked for an engine that isn't there); default is the CPU engine.

func vecsLoad(name string, flat []float32, n, dim int) error {
	if infer.MLXEnabled() {
		return infer.MLXVecsLoad(name, flat, n, dim)
	}
	return cpuVecsLoad(name, flat, n, dim)
}

func vecsTopK(name string, q []float32, k int) ([]int32, []float32, error) {
	if infer.MLXEnabled() {
		return infer.MLXVecsTopK(name, q, k)
	}
	return cpuVecsTopK(name, q, k)
}

func vecsScore(name string, q []float32, n int) ([]float32, error) {
	if infer.MLXEnabled() {
		return infer.MLXVecsScore(name, q, n)
	}
	return cpuVecsScore(name, q, n)
}

func vecEngineLabel() string {
	if infer.MLXEnabled() {
		return "mlx"
	}
	return "cpu"
}

func cpuVecsLoad(name string, flat []float32, n, dim int) error {
	if n*dim != len(flat) {
		return fmt.Errorf("veccpu: %s: %d rows x %d dims != %d floats", name, n, dim, len(flat))
	}
	cpuVecMu.Lock()
	defer cpuVecMu.Unlock()
	if n == 0 { // 0-row load clears the set, same as the MLX engine
		delete(cpuVecSets, name)
		return nil
	}
	cpuVecSets[name] = cpuVecSet{flat: flat, n: n, dim: dim}
	return nil
}

func cpuVecSetGet(name string) (cpuVecSet, error) {
	cpuVecMu.RLock()
	s, ok := cpuVecSets[name]
	cpuVecMu.RUnlock()
	if !ok {
		return cpuVecSet{}, fmt.Errorf("veccpu: set %s not loaded", name)
	}
	return s, nil
}

// cpuScore computes q·row for every row of the set, parallel over row chunks
// (disjoint writes into one shared slice).
func cpuScore(s cpuVecSet, q []float32) []float32 {
	scores := make([]float32, s.n)
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	if workers < 1 || s.n < 256 {
		workers = 1
	}
	chunk := (s.n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, (w+1)*chunk
		if hi > s.n {
			hi = s.n
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				scores[i] = float32(store.Dot(q, s.flat[i*s.dim:(i+1)*s.dim]))
			}
		}(lo, hi)
	}
	wg.Wait()
	return scores
}

// cpuVecsTopK returns the k best rows by dot, indices + scores sorted
// descending — same contract as mlxVecsTopK.
func cpuVecsTopK(name string, q []float32, k int) ([]int32, []float32, error) {
	s, err := cpuVecSetGet(name)
	if err != nil {
		return nil, nil, err
	}
	if k > s.n {
		k = s.n
	}
	if k <= 0 {
		return nil, nil, nil
	}
	scores := cpuScore(s, q)

	// partial selection: size-k min-heap over (score, idx)
	type hit struct {
		idx   int32
		score float32
	}
	heap := make([]hit, 0, k)
	less := func(a, b hit) bool { // heap order: worst on top
		return a.score < b.score
	}
	siftDown := func(i int) {
		for {
			l, r, m := 2*i+1, 2*i+2, i
			if l < len(heap) && less(heap[l], heap[m]) {
				m = l
			}
			if r < len(heap) && less(heap[r], heap[m]) {
				m = r
			}
			if m == i {
				return
			}
			heap[i], heap[m] = heap[m], heap[i]
			i = m
		}
	}
	for i, sc := range scores {
		h := hit{int32(i), sc}
		if len(heap) < k {
			heap = append(heap, h)
			for j := len(heap) - 1; j > 0; {
				p := (j - 1) / 2
				if !less(heap[j], heap[p]) {
					break
				}
				heap[j], heap[p] = heap[p], heap[j]
				j = p
			}
		} else if heap[0].score < sc {
			heap[0] = h
			siftDown(0)
		}
	}
	sort.Slice(heap, func(i, j int) bool { return heap[i].score > heap[j].score })
	idx := make([]int32, len(heap))
	out := make([]float32, len(heap))
	for i, h := range heap {
		idx[i] = h.idx
		out[i] = h.score
	}
	return idx, out, nil
}

// cpuVecsScore returns the full score vector — same contract as mlxVecsScore
// (n is the caller's expected row count, verified loudly).
func cpuVecsScore(name string, q []float32, n int) ([]float32, error) {
	s, err := cpuVecSetGet(name)
	if err != nil {
		return nil, err
	}
	if n != s.n {
		return nil, fmt.Errorf("veccpu: %s has %d rows, caller expects %d", name, s.n, n)
	}
	return cpuScore(s, q), nil
}
