package search

// veccpu_test.go — CPU vector engine: exact parity vs the Go cosine reference
// (fp32 storage, float64-accumulated dot — no fp16 slack needed), the MLX
// engine's clear/error contract, and end-to-end vecStore-vs-SQL ranking parity
// on a real temp db. Runs everywhere; no dylib required.

import (
	"math"
	"oracle/internal/store"
	"sort"
	"testing"
)

func TestCPUVecsTopKParity(t *testing.T) {
	const n, dim, k = 5000, embedDims, 60
	vecs := randUnitVecs(n, dim, 1)
	flat := make([]float32, 0, n*dim)
	for _, v := range vecs {
		flat = append(flat, v...)
	}
	if err := cpuVecsLoad("cpu_parity", flat, n, dim); err != nil {
		t.Fatalf("cpuVecsLoad: %v", err)
	}
	q := randUnitVecs(1, dim, 2)[0]

	type pair struct {
		idx int
		sim float64
	}
	ref := make([]pair, n)
	for i, v := range vecs {
		ref[i] = pair{i, store.Dot(q, v)}
	}
	sort.Slice(ref, func(i, j int) bool { return ref[i].sim > ref[j].sim })

	idx, scores, err := cpuVecsTopK("cpu_parity", q, k)
	if err != nil {
		t.Fatalf("cpuVecsTopK: %v", err)
	}
	if len(idx) != k || len(scores) != k {
		t.Fatalf("got %d idx / %d scores, want %d", len(idx), len(scores), k)
	}
	for i, j := range idx {
		if int(j) != ref[i].idx {
			t.Fatalf("rank %d: idx %d, ref %d", i, j, ref[i].idx)
		}
		if d := math.Abs(float64(scores[i]) - ref[i].sim); d > 1e-6 {
			t.Fatalf("rank %d: score %f vs ref %f", i, scores[i], ref[i].sim)
		}
		if i > 0 && scores[i] > scores[i-1] {
			t.Fatalf("scores not descending at %d", i)
		}
	}
}

func TestCPUVecsKClamp(t *testing.T) {
	vecs := randUnitVecs(3, embedDims, 5)
	flat := make([]float32, 0, 3*embedDims)
	for _, v := range vecs {
		flat = append(flat, v...)
	}
	if err := cpuVecsLoad("cpu_clamp", flat, 3, embedDims); err != nil {
		t.Fatal(err)
	}
	idx, scores, err := cpuVecsTopK("cpu_clamp", vecs[0], 60)
	if err != nil {
		t.Fatal(err)
	}
	if len(idx) != 3 || len(scores) != 3 {
		t.Fatalf("k>n must clamp: got %d", len(idx))
	}
	if idx[0] != 0 || scores[0] < 0.999 {
		t.Fatalf("self-query must rank first: idx %v score %v", idx, scores)
	}
}

func TestCPUVecsLoadClearAndErrors(t *testing.T) {
	if err := cpuVecsLoad("cpu_clear", make([]float32, 4*embedDims), 4, embedDims); err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := cpuVecsLoad("cpu_clear", nil, 0, embedDims); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, _, err := cpuVecsTopK("cpu_clear", make([]float32, embedDims), 5); err == nil {
		t.Fatal("topk on a cleared set must fail loudly")
	}
	if _, _, err := cpuVecsTopK("cpu_never_loaded", make([]float32, embedDims), 5); err == nil {
		t.Fatal("topk on an unknown set must fail loudly")
	}
	if err := cpuVecsLoad("cpu_bad", make([]float32, 7), 4, embedDims); err == nil {
		t.Fatal("size-mismatched load must fail loudly")
	}
	if _, err := cpuVecsScore("cpu_never_loaded", make([]float32, embedDims), 1); err == nil {
		t.Fatal("score on an unknown set must fail loudly")
	}
}

func TestCPUVecsScore(t *testing.T) {
	const n = 300
	vecs := randUnitVecs(n, embedDims, 7)
	flat := make([]float32, 0, n*embedDims)
	for _, v := range vecs {
		flat = append(flat, v...)
	}
	if err := cpuVecsLoad("cpu_score", flat, n, embedDims); err != nil {
		t.Fatal(err)
	}
	q := randUnitVecs(1, embedDims, 8)[0]
	scores, err := cpuVecsScore("cpu_score", q, n)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != n {
		t.Fatalf("got %d scores, want %d", len(scores), n)
	}
	for i, v := range vecs {
		if d := math.Abs(float64(scores[i]) - store.Dot(q, v)); d > 1e-6 {
			t.Fatalf("row %d: score %f vs dot %f", i, scores[i], store.Dot(q, v))
		}
	}
	if _, err := cpuVecsScore("cpu_score", q, n+1); err == nil {
		t.Fatal("row-count mismatch must fail loudly")
	}
}

// TestVecStoreCPUSearchParity builds the store from a real temp db and checks
// cosineTop / paraCosTop / cosineTopAsOf rank identically to the SQL path.
func TestVecStoreCPUSearchParity(t *testing.T) {
	db := store.TestDB(t)
	vecs := randUnitVecs(12, embedDims, 11)
	var ids []int64
	for i := 0; i < 10; i++ {
		id := store.InsertFact(t, db, "cpu parity fact", "fact", "oracle", 1000+float64(i))
		if _, err := db.Exec("INSERT INTO fact_vecs(fact_id, vec) VALUES(?,?)", id, store.VecToBlob(vecs[i])); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	// supersede fact 0 by fact 1 so live/dead/all sets all get coverage
	if _, err := db.Exec("UPDATE facts SET superseded_at=2000, superseded_by=? WHERE id=?", ids[1], ids[0]); err != nil {
		t.Fatal(err)
	}
	// one paraphrase vector for fact 2
	if _, err := db.Exec("INSERT INTO fact_para_vecs(fact_id, vec) VALUES(?,?)", ids[2], store.VecToBlob(vecs[10])); err != nil {
		t.Fatal(err)
	}

	s := &vecStore{}
	if err := s.reload(db); err != nil {
		t.Fatalf("reload: %v", err)
	}
	q := vecs[11]

	gotSim, gotOrder, err := s.cosineTop(q, 5)
	if err != nil {
		t.Fatal(err)
	}
	refSim, refOrder, err := cosineTop(db, q, 0, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotOrder) != len(refOrder) {
		t.Fatalf("order length %d vs %d", len(gotOrder), len(refOrder))
	}
	for i := range refOrder {
		if gotOrder[i] != refOrder[i] {
			t.Fatalf("rank %d: fact %d vs %d", i, gotOrder[i], refOrder[i])
		}
		if d := math.Abs(gotSim[gotOrder[i]] - refSim[refOrder[i]]); d > 1e-6 {
			t.Fatalf("fact %d: sim %f vs %f", gotOrder[i], gotSim[gotOrder[i]], refSim[refOrder[i]])
		}
	}

	gotPara, err := s.paraCosTop(q, 5)
	if err != nil {
		t.Fatal(err)
	}
	refPara, err := paraCosTop(db, q, 0, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotPara) != len(refPara) {
		t.Fatalf("para count %d vs %d", len(gotPara), len(refPara))
	}
	for id, sim := range refPara {
		if math.Abs(gotPara[id]-sim) > 1e-6 {
			t.Fatalf("para fact %d: %f vs %f", id, gotPara[id], sim)
		}
	}

	// as-of frame at T=1000.5: only fact 0 is valid (its superseder becomes
	// valid at 1001) — the frame resurrects a superseded fact, exercising the
	// facts_all set + host-side validity filter.
	asOf := 1000.5
	gotAs, gotAsOrder, err := s.cosineTopAsOf(db, q, asOf, 5)
	if err != nil {
		t.Fatal(err)
	}
	refAs, refAsOrder, err := cosineTop(db, q, asOf, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotAsOrder) != len(refAsOrder) {
		t.Fatalf("as-of order length %d vs %d", len(gotAsOrder), len(refAsOrder))
	}
	for i := range refAsOrder {
		if gotAsOrder[i] != refAsOrder[i] {
			t.Fatalf("as-of rank %d: fact %d vs %d", i, gotAsOrder[i], refAsOrder[i])
		}
		if d := math.Abs(gotAs[gotAsOrder[i]] - refAs[refAsOrder[i]]); d > 1e-6 {
			t.Fatalf("as-of fact %d: sim %f vs %f", gotAsOrder[i], gotAs[gotAsOrder[i]], refAs[refAsOrder[i]])
		}
	}
}
