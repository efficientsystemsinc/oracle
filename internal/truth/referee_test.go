package truth

// Tests for referee.go.

import (
	"database/sql"
	"oracle/internal/store"
	"testing"
)

func addContradictsEdge(t *testing.T, db *sql.DB, src, dst int64) {
	t.Helper()
	if _, err := db.Exec(`INSERT INTO edges(src, dst, type, recorded_at) VALUES(?,?,?,?)`,
		src, dst, "contradicts", 1000.0); err != nil {
		t.Fatal(err)
	}
}

func conflictPairFor(t *testing.T, db *sql.DB, a, b int64) conflictPair {
	t.Helper()
	pairs, err := liveConflicts(db, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pairs {
		if p.A.ID == a && p.B.ID == b {
			return p
		}
	}
	t.Fatalf("pair %d<>%d not in liveConflicts", a, b)
	return conflictPair{}
}

func countRows(t *testing.T, db *sql.DB, q string, args ...any) int {
	t.Helper()
	var n int
	if err := db.QueryRow(q, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// SUPERSEDE_NEWER: newer fact (by valid_from) stamps the older one, the
// contradicts edge becomes a supersedes edge, and the pair is recorded done.
func TestApplySupersedeNewer(t *testing.T) {
	db := store.TestDB(t)
	oldID := store.InsertFact(t, db, "prod is sqlite", "fact", "quasar", 100)
	newID := store.InsertFact(t, db, "prod is postgres", "fact", "quasar", 200)
	// edge src = new fact, dst = old fact (ingest convention)
	addContradictsEdge(t, db, newID, oldID)
	p := conflictPairFor(t, db, newID, oldID)

	if err := applyVerdict(db, p, "SUPERSEDE_NEWER"); err != nil {
		t.Fatal(err)
	}
	var supBy sql.NullInt64
	var supAt sql.NullFloat64
	if err := db.QueryRow(`SELECT superseded_by, superseded_at FROM facts WHERE id = ?`, oldID).
		Scan(&supBy, &supAt); err != nil {
		t.Fatal(err)
	}
	if !supAt.Valid || !supBy.Valid || supBy.Int64 != newID {
		t.Fatalf("old fact not superseded by new: by=%v at=%v", supBy, supAt)
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM facts WHERE id = ? AND superseded_at IS NULL`, newID); n != 1 {
		t.Fatal("newer fact must stay live")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM edges WHERE type='contradicts'`); n != 0 {
		t.Fatal("contradicts edge should be gone")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM edges WHERE type='supersedes' AND src=? AND dst=?`, newID, oldID); n != 1 {
		t.Fatal("supersedes edge missing")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM referee_done WHERE src=? AND dst=? AND verdict='SUPERSEDE_NEWER'`, newID, oldID); n != 1 {
		t.Fatal("referee_done row missing")
	}
	// resume: the pair no longer shows up for refereeing
	if pairs, err := liveConflicts(db, true); err != nil || len(pairs) != 0 {
		t.Fatalf("expected no remaining pairs, got %d err %v", len(pairs), err)
	}
	// both fact rows still exist — never delete a fact
	if n := countRows(t, db, `SELECT COUNT(*) FROM facts`); n != 2 {
		t.Fatal("a fact row was deleted")
	}
}

// SUPERSEDE_NEWER must key on valid_from, not edge direction: if the edge src
// is the OLDER fact, the dst (newer) still wins.
func TestApplySupersedeNewerEdgeDirectionIrrelevant(t *testing.T) {
	db := store.TestDB(t)
	newID := store.InsertFact(t, db, "cap is 10 visits", "fact", "quasar", 500)
	oldID := store.InsertFact(t, db, "cap is 50 visits", "fact", "quasar", 100)
	addContradictsEdge(t, db, oldID, newID) // src = older
	p := conflictPairFor(t, db, oldID, newID)
	if err := applyVerdict(db, p, "SUPERSEDE_NEWER"); err != nil {
		t.Fatal(err)
	}
	var supBy int64
	if err := db.QueryRow(`SELECT superseded_by FROM facts WHERE id = ?`, oldID).Scan(&supBy); err != nil {
		t.Fatal(err)
	}
	if supBy != newID {
		t.Fatalf("older fact superseded_by = %d, want %d", supBy, newID)
	}
}

// DIFFERENT_SCOPE: edge deleted, both facts stay live, verdict recorded.
func TestApplyDifferentScope(t *testing.T) {
	db := store.TestDB(t)
	a := store.InsertFact(t, db, "query path embeds via vertex", "fact", "quasar", 100)
	b := store.InsertFact(t, db, "indexer embeds via m16 on-box", "fact", "quasar", 200)
	addContradictsEdge(t, db, b, a)
	p := conflictPairFor(t, db, b, a)
	if err := applyVerdict(db, p, "DIFFERENT_SCOPE"); err != nil {
		t.Fatal(err)
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL`); n != 2 {
		t.Fatal("both facts must stay live")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM edges`); n != 0 {
		t.Fatal("contradicts edge should be deleted")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM referee_done WHERE verdict='DIFFERENT_SCOPE'`); n != 1 {
		t.Fatal("verdict not recorded")
	}
}

// UNRESOLVED: edge kept (still listed by `oracle conflicts`), facts untouched,
// but the pair is recorded done so referee resumes past it.
func TestApplyUnresolved(t *testing.T) {
	db := store.TestDB(t)
	a := store.InsertFact(t, db, "box is healthy", "status", "quasar", 100)
	b := store.InsertFact(t, db, "box is wedged", "status", "quasar", 100)
	addContradictsEdge(t, db, b, a)
	p := conflictPairFor(t, db, b, a)
	if err := applyVerdict(db, p, "UNRESOLVED"); err != nil {
		t.Fatal(err)
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM edges WHERE type='contradicts'`); n != 1 {
		t.Fatal("contradicts edge must be kept")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL`); n != 2 {
		t.Fatal("facts must be untouched")
	}
	// still visible to `oracle conflicts` (skipDone=false)…
	if pairs, err := liveConflicts(db, false); err != nil || len(pairs) != 1 {
		t.Fatalf("conflicts list: got %d err %v", len(pairs), err)
	}
	// …but skipped on referee resume
	if pairs, err := liveConflicts(db, true); err != nil || len(pairs) != 0 {
		t.Fatalf("resume list: got %d err %v", len(pairs), err)
	}
}

func TestApplyUnknownVerdictFailsLoud(t *testing.T) {
	db := store.TestDB(t)
	a := store.InsertFact(t, db, "x", "fact", "r", 100)
	b := store.InsertFact(t, db, "y", "fact", "r", 200)
	addContradictsEdge(t, db, b, a)
	p := conflictPairFor(t, db, b, a)
	if err := applyVerdict(db, p, "KEEP_BOTH"); err == nil {
		t.Fatal("expected error on unknown verdict")
	}
}

// liveConflicts must exclude pairs where either end is already superseded.
func TestLiveConflictsExcludesDeadEnds(t *testing.T) {
	db := store.TestDB(t)
	a := store.InsertFact(t, db, "a", "fact", "r", 100)
	b := store.InsertFact(t, db, "b", "fact", "r", 200)
	c := store.InsertFact(t, db, "c", "fact", "r", 300)
	addContradictsEdge(t, db, b, a)
	store.Supersede(t, db, a, c, 400) // a no longer live
	if pairs, err := liveConflicts(db, false); err != nil || len(pairs) != 0 {
		t.Fatalf("got %d pairs err %v", len(pairs), err)
	}
}
