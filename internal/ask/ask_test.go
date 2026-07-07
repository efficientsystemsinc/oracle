package ask

// Tests for ask.go.

import (
	"oracle/internal/search"
	"oracle/internal/store"
	"testing"
)

func TestCitedIDs(t *testing.T) {
	seen := map[int64]search.FactOut{12: {ID: 12}, 7: {ID: 7}, 40: {ID: 40}}
	answer := "atlas01 is PROD [12], confirmed twice [12] by the runbook [7]. Unknown source [999]."
	got := citedIDs(answer, seen)
	if len(got) != 2 || got[0] != 12 || got[1] != 7 {
		t.Errorf("want [12 7] (deduped, in order, only surfaced ids), got %v", got)
	}
	if got := citedIDs("no citations here", seen); len(got) != 0 {
		t.Errorf("want none, got %v", got)
	}
}

func TestReinforceFacts(t *testing.T) {
	db := store.TestDB(t)
	a := store.InsertFact(t, db, "asyncpg needs statement_cache_size=0 behind pgbouncer", "gotcha", "quasar", 100)
	b := store.InsertFact(t, db, "near mass cap", "fact", "quasar", 100)
	if _, err := db.Exec("UPDATE facts SET mass = 2.95 WHERE id = ?", b); err != nil {
		t.Fatal(err)
	}

	search.ReinforceFacts(db, []int64{a, b})
	var mass float64
	var useCount int
	if err := db.QueryRow("SELECT mass, use_count FROM facts WHERE id = ?", a).Scan(&mass, &useCount); err != nil {
		t.Fatal(err)
	}
	if mass != 1.15 || useCount != 1 {
		t.Errorf("want mass 1.15 use_count 1, got %v %d", mass, useCount)
	}
	if err := db.QueryRow("SELECT mass FROM facts WHERE id = ?", b).Scan(&mass); err != nil {
		t.Fatal(err)
	}
	if mass != 3.0 {
		t.Errorf("mass must cap at 3.0, got %v", mass)
	}

	search.ReinforceFacts(db, nil) // empty ids: no-op
}
