package truth

// Tests for sweep.go.

import (
	"database/sql"
	"oracle/internal/store"
	"testing"
)

// insertVec writes a unit vector for a fact directly (no LLM in tests).
func insertVec(t *testing.T, db *sql.DB, factID int64, v []float32) {
	t.Helper()
	if _, err := db.Exec("INSERT INTO fact_vecs(fact_id, vec) VALUES(?,?)",
		factID, store.VecToBlob(store.Normalize(v))); err != nil {
		t.Fatal(err)
	}
}

func linkEnt(t *testing.T, db *sql.DB, entID, factID int64) {
	t.Helper()
	if _, err := db.Exec("INSERT INTO fact_entities(fact_id, entity_id) VALUES(?,?)", factID, entID); err != nil {
		t.Fatal(err)
	}
}

func TestSweepCandidates(t *testing.T) {
	db := store.TestDB(t)
	if _, err := db.Exec(sweepSchema); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("INSERT INTO entities(id, name, display) VALUES(1,'atlas01','atlas01')"); err != nil {
		t.Fatal(err)
	}
	day := 86400.0

	// similar vecs, same kind, 10 days apart -> should pair
	a := store.InsertFact(t, db, "api runs on box-a", "fact", "quasar", 100*day)
	b := store.InsertFact(t, db, "api runs on box-b", "fact", "quasar", 110*day)
	// same kind, similar vec, but only 1 day from a -> gap too small vs a
	c := store.InsertFact(t, db, "api runs on box-c", "fact", "quasar", 101*day)
	// different kind, similar vec, distant date -> excluded
	d := store.InsertFact(t, db, "decide to move api", "decision", "quasar", 130*day)
	// same kind, distant date, dissimilar vec -> excluded
	e := store.InsertFact(t, db, "ci uses runner-x", "fact", "quasar", 140*day)
	// same kind, similar vec, distant date, but already edged with a -> excluded
	f := store.InsertFact(t, db, "api runs on box-f", "fact", "quasar", 150*day)

	sim := []float32{1, 0.1, 0}   // near-identical direction family
	sim2 := []float32{1, 0.15, 0} // cosine with sim ~0.999
	far := []float32{0, 0, 1}     // orthogonal
	insertVec(t, db, a, sim)
	insertVec(t, db, b, sim2)
	insertVec(t, db, c, sim)
	insertVec(t, db, d, sim)
	insertVec(t, db, e, far)
	insertVec(t, db, f, sim)
	for _, id := range []int64{a, b, c, d, e, f} {
		linkEnt(t, db, 1, id)
	}
	if _, err := db.Exec("INSERT INTO edges(src, dst, type, recorded_at) VALUES(?,?,?,0)", f, a, "supersedes"); err != nil {
		t.Fatal(err)
	}

	pairs, err := sweepCandidates(db, 6, 2000)
	if err != nil {
		t.Fatal(err)
	}
	got := map[[2]int64]sweepPair{}
	for _, p := range pairs {
		got[p.key()] = p
		if p.OldVF >= p.NewVF {
			t.Fatalf("pair %v: Old must have earlier valid_from", p)
		}
	}
	// expected pairs: (a,b), (b,c)? c is 101d, b is 110d -> 9d gap same kind similar -> pair.
	// (a,c) gap 1d -> no. (x,d) kind differs -> no. (x,e) cosine low -> no. (a,f) edged -> no.
	// (b,f) 40d gap similar same kind -> pair. (c,f) 49d -> pair.
	want := [][2]int64{{a, b}, {b, c}, {b, f}, {c, f}}
	for _, w := range want {
		k := w
		if k[0] > k[1] {
			k[0], k[1] = k[1], k[0]
		}
		if _, ok := got[k]; !ok {
			t.Errorf("expected pair %v missing (got %v)", k, pairs)
		}
	}
	bad := [][2]int64{{a, c}, {a, d}, {a, e}, {a, f}}
	for _, w := range bad {
		k := w
		if k[0] > k[1] {
			k[0], k[1] = k[1], k[0]
		}
		if _, ok := got[k]; ok {
			t.Errorf("pair %v should have been excluded", k)
		}
	}

	// per-entity cap keeps the highest-similarity pairs
	capped, err := sweepCandidates(db, 2, 2000)
	if err != nil {
		t.Fatal(err)
	}
	if len(capped) != 2 {
		t.Fatalf("per-entity cap: got %d pairs, want 2", len(capped))
	}

	// sweep_done exclusion: mark (a,b) judged, it must disappear
	k := sweepPair{Old: a, New: b}.key()
	if _, err := db.Exec("INSERT INTO sweep_done(a,b,verdict,judged_at) VALUES(?,?,?,0)", k[0], k[1], "nothing"); err != nil {
		t.Fatal(err)
	}
	pairs2, err := sweepCandidates(db, 6, 2000)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pairs2 {
		if p.key() == k {
			t.Fatalf("sweep_done pair %v resurfaced", k)
		}
	}
	if len(pairs2) != len(pairs)-1 {
		t.Fatalf("expected %d pairs after sweep_done, got %d", len(pairs)-1, len(pairs2))
	}
}

func TestSweepApply(t *testing.T) {
	db := store.TestDB(t)
	if _, err := db.Exec(sweepSchema); err != nil {
		t.Fatal(err)
	}
	day := 86400.0
	old := store.InsertFact(t, db, "api runs on box-a", "fact", "quasar", 100*day)
	new_ := store.InsertFact(t, db, "api runs on box-b", "fact", "quasar", 120*day)
	p := sweepPair{Old: old, New: new_, OldVF: 100 * day, NewVF: 120 * day}
	if err := sweepApply(db, p, "supersedes"); err != nil {
		t.Fatal(err)
	}
	var supBy sql.NullInt64
	var supAt sql.NullFloat64
	if err := db.QueryRow("SELECT superseded_by, superseded_at FROM facts WHERE id=?", old).Scan(&supBy, &supAt); err != nil {
		t.Fatal(err)
	}
	if !supBy.Valid || supBy.Int64 != new_ || !supAt.Valid {
		t.Fatalf("old fact not superseded: by=%v at=%v", supBy, supAt)
	}
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM edges WHERE src=? AND dst=? AND type='supersedes'", new_, old).Scan(&n); err != nil || n != 1 {
		t.Fatalf("supersedes edge missing (n=%d err=%v)", n, err)
	}
	if err := db.QueryRow("SELECT COUNT(*) FROM sweep_done WHERE a=? AND b=?", min(old, new_), max(old, new_)).Scan(&n); err != nil || n != 1 {
		t.Fatalf("sweep_done row missing (n=%d err=%v)", n, err)
	}
}
