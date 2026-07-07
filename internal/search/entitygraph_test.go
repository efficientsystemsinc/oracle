package search

import (
	"oracle/internal/kb"
	"oracle/internal/store"
	"testing"
)

// A respelled mention must fold onto the existing entity (recording an alias),
// not mint a duplicate — this is what made 28 post-hoc merges necessary.
func TestLinkEntitiesFoldsVariants(t *testing.T) {
	db := store.TestDB(t)
	canonical := kb.InsertEntity(t, db, "upload-delta")
	f := store.InsertFact(t, db, "delta uploads are pinned to base index", "fact", "quasar", 100)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := linkEntities(tx, f, []string{"upload_delta"}, 200); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}

	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM entities").Scan(&n); err != nil || n != 1 {
		t.Errorf("variant must fold, not create: %d entities (err %v)", n, err)
	}
	var linked int64
	if err := db.QueryRow("SELECT entity_id FROM fact_entities WHERE fact_id = ?", f).Scan(&linked); err != nil || linked != canonical {
		t.Errorf("fact should link to canonical entity %d, got %d (err %v)", canonical, linked, err)
	}
	// the observed spelling becomes an exact alias for next time
	if id, _, ok := kb.ResolveEntityExact(db, "upload_delta"); !ok || id != canonical {
		t.Errorf("observed spelling should now resolve exactly, got %d ok=%v", id, ok)
	}
}

// After `oracle merge`, ingest mentioning the merged-away name must not
// resurrect it — the old linkEntities inserted by exact name and would have.
func TestMergeSticksThroughIngest(t *testing.T) {
	db := store.TestDB(t)
	winner := kb.InsertEntity(t, db, "quasar api")
	kb.InsertEntity(t, db, "quasarapi2") // distinct name, avoid variant overlap with winner
	if err := kb.MergeEntities(db, "quasar api", "quasarapi2"); err != nil {
		t.Fatal(err)
	}
	f := store.InsertFact(t, db, "the api restarted", "status", "quasar", 100)
	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := linkEntities(tx, f, []string{"quasarapi2"}, 200); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM entities").Scan(&n); err != nil || n != 1 {
		t.Errorf("merged-away name must not be resurrected: %d entities (err %v)", n, err)
	}
	var linked int64
	if err := db.QueryRow("SELECT entity_id FROM fact_entities WHERE fact_id = ?", f).Scan(&linked); err != nil || linked != winner {
		t.Errorf("mention of merged name should link to winner %d, got %d (err %v)", winner, linked, err)
	}
}

// After a merge, the loser's name must resolve to the winner everywhere —
// including entityView, which historically had its own alias-blind lookup.
func TestEntityViewResolvesMergedAlias(t *testing.T) {
	db := store.TestDB(t)
	winner := kb.InsertEntity(t, db, "quasar api")
	kb.InsertEntity(t, db, "quasar-api")
	f := store.InsertFact(t, db, "the api serves on :4141", "fact", "quasar", 100)
	if _, err := db.Exec(`INSERT INTO fact_entities(fact_id, entity_id) VALUES(?,?)`, f, winner); err != nil {
		t.Fatal(err)
	}
	if err := kb.MergeEntities(db, "quasar api", "quasar-api"); err != nil {
		t.Fatal(err)
	}
	v, err := EntityView(db, "quasar-api", 10)
	if err != nil {
		t.Fatal(err)
	}
	if v["entity"] != "quasar api" {
		t.Errorf("loser name should resolve to winner via alias, got %v", v["entity"])
	}
	if len(v["facts"].([]FactOut)) != 1 {
		t.Errorf("winner's facts should be visible via alias lookup")
	}
}
