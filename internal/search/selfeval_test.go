package search

// Tests for selfeval.go.

import (
	"encoding/json"
	"oracle/internal/store"
	"testing"
)

func TestSupersessionPairs(t *testing.T) {
	db := store.TestDB(t)
	old := store.InsertFact(t, db, "api on box-a", "fact", "quasar", 1000)
	new_ := store.InsertFact(t, db, "api on box-b", "fact", "quasar", 90000) // >1h later
	store.Supersede(t, db, old, new_, 100000)
	// too-close pair must be excluded (as-of midpoint meaningless)
	c1 := store.InsertFact(t, db, "flag on", "status", "quasar", 5000)
	c2 := store.InsertFact(t, db, "flag off", "status", "quasar", 5010)
	store.Supersede(t, db, c1, c2, 100000)

	pairs, err := supersessionPairs(db, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 || pairs[0].oldID != old || pairs[0].newID != new_ {
		t.Fatalf("want exactly the well-separated pair, got %+v", pairs)
	}
	if pairs[0].oldStmt != "api on box-a" {
		t.Errorf("pair should carry the old statement, got %q", pairs[0].oldStmt)
	}
}

func TestLogAskTraceStoresCitations(t *testing.T) {
	db := store.TestDB(t)
	LogAskTrace(db, "why box-b?", []string{`search({"q":"box"})`}, []int64{7, 12})
	var res string
	if err := db.QueryRow(`SELECT results FROM traces WHERE kind='ask'`).Scan(&res); err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Calls []string `json:"calls"`
		Cited []int64  `json:"cited"`
	}
	if err := json.Unmarshal([]byte(res), &payload); err != nil {
		t.Fatalf("ask trace must be structured JSON now: %v (%s)", err, res)
	}
	if len(payload.Cited) != 2 || payload.Cited[0] != 7 || len(payload.Calls) != 1 {
		t.Errorf("round-trip mismatch: %+v", payload)
	}
}
