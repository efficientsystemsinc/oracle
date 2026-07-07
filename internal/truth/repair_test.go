package truth

// Tests for repair.go.

import (
	"oracle/internal/store"
	"testing"
)

func TestRepairVerdictApplication(t *testing.T) {
	db := store.TestDB(t)
	if err := applyRepairSchema(db); err != nil {
		t.Fatal(err)
	}

	old1 := store.InsertFact(t, db, "prod DB is sqlite", "fact", "quasar", 100)
	new1 := store.InsertFact(t, db, "prod DB is postgres", "fact", "quasar", 200)
	store.Supersede(t, db, old1, new1, 200)
	mustExec := func(q string, args ...any) {
		t.Helper()
		if _, err := db.Exec(q, args...); err != nil {
			t.Fatal(err)
		}
	}
	mustExec(`INSERT INTO edges(src,dst,type,recorded_at) VALUES(?,?,'supersedes',200)`, new1, old1)

	old2 := store.InsertFact(t, db, "box A serves the planner", "fact", "quasar", 100)
	new2 := store.InsertFact(t, db, "box B serves the embedder", "fact", "quasar", 200)
	store.Supersede(t, db, old2, new2, 200)
	mustExec(`INSERT INTO edges(src,dst,type,recorded_at) VALUES(?,?,'supersedes',200)`, new2, old2)

	old3 := store.InsertFact(t, db, "reranker is enabled", "fact", "quasar", 300)
	new3 := store.InsertFact(t, db, "reranker is disabled", "fact", "quasar", 200) // OLDER than old3
	store.Supersede(t, db, old3, new3, 350)
	mustExec(`INSERT INTO edges(src,dst,type,recorded_at) VALUES(?,?,'supersedes',350)`, new3, old3)

	pairs, err := loadRepairPairs(db, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 3 {
		t.Fatalf("want 3 pairs, got %d", len(pairs))
	}

	// Fake verdict set — no LLM in tests.
	fake := map[int64]string{
		old1: "UPHOLD",
		old2: "REOPEN",
		old3: "REOPEN_CONTRADICT",
	}
	for _, p := range pairs {
		if err := applyRepairVerdict(db, p, fake[p.OldID]); err != nil {
			t.Fatal(err)
		}
	}

	// UPHOLD: old1 stays closed, edge intact.
	var supBy *int64
	var supAt *float64
	if err := db.QueryRow(`SELECT superseded_by, superseded_at FROM facts WHERE id=?`, old1).Scan(&supBy, &supAt); err != nil {
		t.Fatal(err)
	}
	if supBy == nil || *supBy != new1 || supAt == nil {
		t.Fatalf("UPHOLD must leave old1 closed, got by=%v at=%v", supBy, supAt)
	}
	if n := countEdges(t, db, new1, old1, "supersedes"); n != 1 {
		t.Fatalf("UPHOLD must keep supersedes edge, got %d", n)
	}

	// REOPEN: old2 live again, supersedes edge gone, no contradicts edge.
	if err := db.QueryRow(`SELECT superseded_by, superseded_at FROM facts WHERE id=?`, old2).Scan(&supBy, &supAt); err != nil {
		t.Fatal(err)
	}
	if supBy != nil || supAt != nil {
		t.Fatalf("REOPEN must clear supersession on old2, got by=%v at=%v", supBy, supAt)
	}
	if n := countEdges(t, db, new2, old2, "supersedes"); n != 0 {
		t.Fatalf("REOPEN must delete supersedes edge, got %d", n)
	}
	if n := countEdges(t, db, new2, old2, "contradicts"); n != 0 {
		t.Fatalf("REOPEN must NOT add contradicts edge, got %d", n)
	}

	// REOPEN_CONTRADICT: old3 live, supersedes gone, contradicts(new->old) added.
	if err := db.QueryRow(`SELECT superseded_by, superseded_at FROM facts WHERE id=?`, old3).Scan(&supBy, &supAt); err != nil {
		t.Fatal(err)
	}
	if supBy != nil || supAt != nil {
		t.Fatalf("REOPEN_CONTRADICT must clear supersession on old3, got by=%v at=%v", supBy, supAt)
	}
	if n := countEdges(t, db, new3, old3, "supersedes"); n != 0 {
		t.Fatalf("REOPEN_CONTRADICT must delete supersedes edge, got %d", n)
	}
	if n := countEdges(t, db, new3, old3, "contradicts"); n != 1 {
		t.Fatalf("REOPEN_CONTRADICT must add contradicts edge, got %d", n)
	}

	// All three marked done; a reload finds nothing pending (resumability).
	st, err := repairStatus(db)
	if err != nil {
		t.Fatal(err)
	}
	if st["UPHOLD"] != 1 || st["REOPEN"] != 1 || st["REOPEN_CONTRADICT"] != 1 {
		t.Fatalf("bad repair_done counts: %v", st)
	}
	left, err := loadRepairPairs(db, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(left) != 0 {
		t.Fatalf("all pairs audited, want 0 pending, got %d", len(left))
	}
}

func TestRepairInvalidVerdictRejected(t *testing.T) {
	db := store.TestDB(t)
	if err := applyRepairSchema(db); err != nil {
		t.Fatal(err)
	}
	oldID := store.InsertFact(t, db, "a", "fact", "r", 100)
	newID := store.InsertFact(t, db, "b", "fact", "r", 200)
	store.Supersede(t, db, oldID, newID, 200)
	pairs, err := loadRepairPairs(db, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := applyRepairVerdict(db, pairs[0], "MAYBE"); err == nil {
		t.Fatal("invalid verdict must be rejected loudly")
	}
	// Nothing written: still pending.
	left, err := loadRepairPairs(db, 0)
	if err != nil || len(left) != 1 {
		t.Fatalf("invalid verdict must not consume the pair: err=%v left=%d", err, len(left))
	}
}

func countEdges(t *testing.T, db store.DBQ, src, dst int64, typ string) int {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM edges WHERE src=? AND dst=? AND type=?`,
		src, dst, typ).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}
