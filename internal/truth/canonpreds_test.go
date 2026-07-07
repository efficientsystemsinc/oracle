package truth

// Tests for canonpreds.go.

import (
	"database/sql"
	"oracle/internal/kb"
	"oracle/internal/store"
	"testing"
)

func insertPredicate(t *testing.T, db *sql.DB, name string, seen int64) int64 {
	t.Helper()
	res, err := db.Exec("INSERT INTO predicates(name, seen_count) VALUES(?,?)", name, seen)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := res.LastInsertId()
	return id
}

func insertTripleP(t *testing.T, db *sql.DB, subj, pid, obj int64) {
	t.Helper()
	if _, err := db.Exec(`INSERT INTO triples(fact_id, subject_id, predicate_id, object_id, recorded_at)
		VALUES(1,?,?,?,0)`, subj, pid, obj); err != nil {
		t.Fatal(err)
	}
}

func TestResolveCanonChains(t *testing.T) {
	out := resolveCanonChains(map[string]string{
		"hosted_on":   "deployed_on", // chain: hosted_on -> deployed_on -> runs_on
		"deployed_on": "runs_on",
		"runs_on":     "runs_on", // self-map: dropped
		"uses":        "Uses It", // messy canonical: cleaned to snake_case, then self... no: uses -> uses_it
		"a":           "b",       // cycle a<->b: a keeps itself
		"b":           "a",
	})
	if out["hosted_on"] != "runs_on" || out["deployed_on"] != "runs_on" {
		t.Errorf("chain not resolved: %v", out)
	}
	if _, ok := out["runs_on"]; ok {
		t.Errorf("self-map survived: %v", out)
	}
	if out["uses"] != "uses_it" {
		t.Errorf("canonical not cleaned: %v", out)
	}
	// cycles must not produce an infinite loop or a mapping to a folded-away name
	for k, v := range out {
		if k == v {
			t.Errorf("self map %q leaked", k)
		}
	}
}

func TestApplyCanonMappingFolds(t *testing.T) {
	db := store.TestDB(t)
	s := kb.InsertEntity(t, db, "quasar")
	o := kb.InsertEntity(t, db, "atlas01")
	runsOn := insertPredicate(t, db, "runs_on", 10)
	hosted := insertPredicate(t, db, "hosted_on", 3)
	deployed := insertPredicate(t, db, "deployed_on_the_h100_box", 2)
	uses := insertPredicate(t, db, "uses", 7)
	insertTripleP(t, db, s, hosted, o)
	insertTripleP(t, db, s, deployed, o)
	insertTripleP(t, db, s, runsOn, o)
	insertTripleP(t, db, s, uses, o)

	folded, err := applyCanonMapping(db, map[string]string{
		"hosted_on":                "runs_on",
		"deployed_on_the_h100_box": "runs_on",
		"never_existed":            "runs_on",  // vanished predicate: skipped, not an error
		"depends_on":               "requires", // canonical doesn't exist yet: created
	})
	if err != nil {
		t.Fatal(err)
	}
	if folded != 2 {
		t.Errorf("folded = %d, want 2", folded)
	}
	// triples repointed
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM triples WHERE predicate_id = ?", runsOn).Scan(&n); err != nil || n != 3 {
		t.Errorf("runs_on triples = %d (err %v), want 3", n, err)
	}
	// seen_counts folded: 10 + 3 + 2
	var seen int64
	if err := db.QueryRow("SELECT seen_count FROM predicates WHERE id = ?", runsOn).Scan(&seen); err != nil || seen != 15 {
		t.Errorf("runs_on seen_count = %d (err %v), want 15", seen, err)
	}
	// orphans deleted, untouched predicate intact
	for _, gone := range []string{"hosted_on", "deployed_on_the_h100_box"} {
		var id int64
		if db.QueryRow("SELECT id FROM predicates WHERE name = ?", gone).Scan(&id) != sql.ErrNoRows {
			t.Errorf("%s should be deleted", gone)
		}
	}
	var usesSeen int64
	if err := db.QueryRow("SELECT seen_count FROM predicates WHERE name = 'uses'").Scan(&usesSeen); err != nil || usesSeen != 7 {
		t.Errorf("uses disturbed: %d (err %v)", usesSeen, err)
	}
	// aliases recorded
	var pid int64
	if err := db.QueryRow("SELECT predicate_id FROM predicate_aliases WHERE alias = 'hosted_on'").Scan(&pid); err != nil || pid != runsOn {
		t.Errorf("alias hosted_on -> %d (err %v), want %d", pid, err, runsOn)
	}
	// self-map must be rejected loudly
	if _, err := applyCanonMapping(db, map[string]string{"uses": "uses"}); err == nil {
		t.Error("self-map should error")
	}
}

// storeTriples must route a folded-away predicate name through predicate_aliases
// instead of re-creating the deleted predicate row.
func TestStoreTriplesConsultsAliases(t *testing.T) {
	db := store.TestDB(t)
	runsOn := insertPredicate(t, db, "runs_on", 5)
	if _, err := db.Exec("INSERT INTO predicate_aliases(alias, predicate_id) VALUES('hosted_on', ?)", runsOn); err != nil {
		t.Fatal(err)
	}
	if err := kb.StoreTriples(db, 1, []kb.Triple{{Subject: "quasar", Predicate: "hosted on", Object: "atlas01"}}, 100); err != nil {
		t.Fatal(err)
	}
	var id int64
	if db.QueryRow("SELECT id FROM predicates WHERE name = 'hosted_on'").Scan(&id) != sql.ErrNoRows {
		t.Error("folded predicate resurrected")
	}
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM triples WHERE predicate_id = ?", runsOn).Scan(&n); err != nil || n != 1 {
		t.Errorf("triple not routed to canonical: n=%d err=%v", n, err)
	}
	var seen int64
	if err := db.QueryRow("SELECT seen_count FROM predicates WHERE id = ?", runsOn).Scan(&seen); err != nil || seen != 6 {
		t.Errorf("seen_count = %d, want 6", seen)
	}
}
