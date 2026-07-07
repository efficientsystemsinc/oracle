package search

// Tests for embed_local.go.

import (
	"oracle/internal/store"
	"strings"
	"testing"
)

// fakeLocal is a deterministic stand-in for the ONNX embedLocal.
func fakeLocal(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, embedDims)
		v[i%embedDims] = 1
		out[i] = v
	}
	return out, nil
}

func TestEmbedTextsRoutesToLocalWhenFlagOn(t *testing.T) {
	orig := embedLocalFn
	t.Cleanup(func() { embedLocalFn = orig })

	called := 0
	embedLocalFn = func(texts []string) ([][]float32, error) {
		called++
		return fakeLocal(texts)
	}

	t.Setenv("ORACLE_LOCAL_EMBED", "1")
	vecs, err := embedTexts([]string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 || len(vecs) != 2 || len(vecs[0]) != embedDims {
		t.Fatalf("local backend not used: called=%d vecs=%d", called, len(vecs))
	}

	// Flag off: must NOT touch the local backend (remote path; fails fast on
	// the bogus URL instead).
	t.Setenv("ORACLE_LOCAL_EMBED", "")
	t.Setenv("ORACLE_EMBED_URL", "http://127.0.0.1:0/embeddings")
	t.Setenv("ORACLE_EMBED_KEY", "test")
	_, _ = embedTexts([]string{"a"})
	if called != 1 {
		t.Fatalf("remote path leaked into local backend: called=%d", called)
	}
}

func TestVecsTableSelection(t *testing.T) {
	t.Setenv("ORACLE_LOCAL_EMBED", "")
	if factVecsTable() != "fact_vecs" || paraVecsTable() != "fact_para_vecs" {
		t.Fatal("flag off must use remote tables")
	}
	t.Setenv("ORACLE_LOCAL_EMBED", "1")
	if factVecsTable() != "fact_vecs_local" || paraVecsTable() != "fact_para_vecs_local" {
		t.Fatal("flag on must use _local tables")
	}
}

func TestCheckLocalVecsGuardsEmptyCorpus(t *testing.T) {
	db := store.TestDB(t)

	t.Setenv("ORACLE_LOCAL_EMBED", "1")
	err := checkLocalVecs(db)
	if err == nil || !strings.Contains(err.Error(), "reembed") {
		t.Fatalf("empty fact_vecs_local with flag on must error loudly, got %v", err)
	}
	// cosineTop must surface the same guard.
	if _, _, err := cosineTop(db, make([]float32, embedDims), 0, 5); err == nil {
		t.Fatal("cosineTop must refuse to search an unmigrated corpus")
	}

	if _, err := db.Exec("INSERT INTO fact_vecs_local(fact_id, vec) VALUES(1, ?)",
		store.VecToBlob(make([]float32, embedDims))); err != nil {
		t.Fatal(err)
	}
	if err := checkLocalVecs(db); err != nil {
		t.Fatalf("populated _local table must pass: %v", err)
	}

	t.Setenv("ORACLE_LOCAL_EMBED", "")
	if err := checkLocalVecs(db); err != nil {
		t.Fatalf("flag off must never guard: %v", err)
	}
}

func TestReembedWritesLocalTablesAndResumes(t *testing.T) {
	db := store.TestDB(t)
	orig := embedLocalFn
	t.Cleanup(func() { embedLocalFn = orig })
	embedded := 0
	embedLocalFn = func(texts []string) ([][]float32, error) {
		embedded += len(texts)
		return fakeLocal(texts)
	}
	t.Setenv("ORACLE_LOCAL_EMBED", "1")

	a := store.InsertFact(t, db, "fact one", "decision", "r", 1000)
	b := store.InsertFact(t, db, "fact two", "gotcha", "r", 1000)
	dead := store.InsertFact(t, db, "old fact", "status", "r", 900)
	store.Supersede(t, db, dead, b, 1001)
	if _, err := db.Exec("INSERT INTO fact_paraphrases(fact_id, text) VALUES(?, 'para one')", a); err != nil {
		t.Fatal(err)
	}

	if err := Reembed(db, 64); err != nil {
		t.Fatal(err)
	}
	var nf, np int
	must2(t, db.QueryRow("SELECT COUNT(*) FROM fact_vecs_local").Scan(&nf))
	must2(t, db.QueryRow("SELECT COUNT(*) FROM fact_para_vecs_local").Scan(&np))
	if nf != 2 || np != 1 { // live facts only, one paraphrase
		t.Fatalf("got %d fact vecs, %d para vecs; want 2, 1", nf, np)
	}

	// Resumable: a second run embeds nothing new.
	before := embedded
	if err := Reembed(db, 64); err != nil {
		t.Fatal(err)
	}
	if embedded != before {
		t.Fatalf("reembed re-embedded already-done rows: %d -> %d", before, embedded)
	}

	// With the corpus migrated, search reads _local and works end to end.
	qv, err := embedTexts([]string{"fact one"})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := cosineTop(db, qv[0], 0, 5); err != nil {
		t.Fatalf("cosineTop over _local corpus: %v", err)
	}
}

func must2(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
