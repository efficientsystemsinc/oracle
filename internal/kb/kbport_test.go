package kb

// Tests for kbport.go.

import (
	"database/sql"
	"oracle/internal/store"
	"strings"
	"testing"
)

func TestValidEntityName(t *testing.T) {
	valid := []string{"atlas01", "quasar-prod-flex", "text-embedding-3-large", "gpt-5.5", "us east4 gke cluster"}
	for _, n := range valid {
		if !ValidEntityName(n) {
			t.Errorf("%q should be valid", n)
		}
	}
	invalid := []string{
		"",
		"src/extract.go",                    // path fragment
		`the "main" box`,                    // quotes
		"we decided to move the api to gke", // sentence (>3 spaces)
		strings.Repeat("x", 49),             // too long
	}
	for _, n := range invalid {
		if ValidEntityName(n) {
			t.Errorf("%q should be rejected", n)
		}
	}
}

func TestVariantKey(t *testing.T) {
	for _, v := range []string{"upload_delta", "Upload-Delta", "upload delta", "upload.delta", "UPLOADDELTA"} {
		if VariantKey(v) != "uploaddelta" {
			t.Errorf("variantKey(%q) = %q", v, VariantKey(v))
		}
	}
	if VariantKey("upload-deltas") == "uploaddelta" {
		t.Error("distinct names must not collide")
	}
}

// Two existing entities sharing a variant key: an incoming variant is
// ambiguous, so it must create a new entity rather than guess.
func TestResolveOrCreateAmbiguousVariant(t *testing.T) {
	db := store.TestDB(t)
	InsertEntity(t, db, "a-b")
	InsertEntity(t, db, "ab")
	id, err := ResolveOrCreateEntity(db, "a_b", "", 100)
	if err != nil {
		t.Fatal(err)
	}
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM entities").Scan(&n); err != nil || n != 3 {
		t.Errorf("ambiguous variant must create new entity (got %d entities, id %d, err %v)", n, id, err)
	}
}

// The enrich path (typed entities, triple subjects/objects) folds the same way.
func TestUpsertEntityFoldsAndTypes(t *testing.T) {
	db := store.TestDB(t)
	canonical := InsertEntity(t, db, "upload-delta")
	id, err := UpsertEntityByName(db, "Upload Delta", "service", 200)
	if err != nil || id != canonical {
		t.Fatalf("enrich mention should fold onto %d, got %d (err %v)", canonical, id, err)
	}
	var etype string
	if err := db.QueryRow("SELECT COALESCE(etype,'') FROM entities WHERE id = ?", id).Scan(&etype); err != nil || etype != "service" {
		t.Errorf("etype should be set on fold, got %q (err %v)", etype, err)
	}
}

func insertTriple(t *testing.T, db *sql.DB, subj, obj int64, pred string) {
	t.Helper()
	if _, err := db.Exec(`INSERT INTO predicates(name, seen_count) VALUES(?,1)
		ON CONFLICT(name) DO NOTHING`, pred); err != nil {
		t.Fatal(err)
	}
	var pid int64
	if err := db.QueryRow(`SELECT id FROM predicates WHERE name = ?`, pred).Scan(&pid); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO triples(fact_id, subject_id, predicate_id, object_id, recorded_at)
		VALUES(0,?,?,?,0)`, subj, pid, obj); err != nil {
		t.Fatal(err)
	}
}

func insertCoMention(t *testing.T, db *sql.DB, a, b int64) {
	t.Helper()
	lo, hi := min(a, b), max(a, b)
	if _, err := db.Exec(`INSERT INTO entity_edges(a, b, count, last_seen)
		VALUES(?,?,3,0)`, lo, hi); err != nil {
		t.Fatal(err)
	}
}

// quasar -runs_on-> atlas01 (triple); quasar co-mentions pgbouncer (no triple).
// Traversal must reach both: pgbouncer only via the co-mention fallback, and
// atlas01 must arrive as the typed link, not be shadowed by a co-mention.
func TestTraverseCoMentionFallback(t *testing.T) {
	db := store.TestDB(t)
	quasar := InsertEntity(t, db, "quasar")
	atlas01 := InsertEntity(t, db, "atlas01")
	pgb := InsertEntity(t, db, "pgbouncer")
	insertTriple(t, db, quasar, atlas01, "runs_on")
	insertCoMention(t, db, quasar, pgb)
	insertCoMention(t, db, quasar, atlas01) // duplicate path; typed link should win

	v, err := Traverse(db, "quasar", 2, 60)
	if err != nil {
		t.Fatal(err)
	}
	links := v["links"].([]map[string]any)
	byTarget := map[string]string{}
	for _, l := range links {
		byTarget[l["to"].(string)] = l["predicate"].(string)
	}
	if byTarget["atlas01"] != "runs_on" {
		t.Errorf("atlas01 should be reached via typed triple, got %v", byTarget)
	}
	if byTarget["pgbouncer"] != "co_mentioned" {
		t.Errorf("pgbouncer should be reached via co-mention fallback, got %v", byTarget)
	}
	if v["reached"].(int) != 2 {
		t.Errorf("expected 2 reached, got %v", v["reached"])
	}
}

func TestTraverseRespectsLimit(t *testing.T) {
	db := store.TestDB(t)
	hub := InsertEntity(t, db, "hub")
	for i := 0; i < 10; i++ {
		other := InsertEntity(t, db, string(rune('a'+i))+"-node")
		insertCoMention(t, db, hub, other)
	}
	v, err := Traverse(db, "hub", 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(v["links"].([]map[string]any)); n > 4 {
		t.Errorf("limit 4 exceeded: %d links", n)
	}
}
