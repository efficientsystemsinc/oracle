package search

// Tests for introspect.go.

import (
	"oracle/internal/store"
	"strings"
	"testing"
)

func TestOptimizeRelabelsJunkRepos(t *testing.T) {
	db := store.TestDB(t)
	junk := "2026-06-20-install-this-skill-https-github-com"
	store.InsertFact(t, db, "some scratch-session fact", "fact", junk, 100)
	keep := store.InsertFact(t, db, "real repo fact", "fact", "quasar", 100)
	if _, err := db.Exec(`INSERT INTO files(path, source, repo, offset, mtime) VALUES('/x.jsonl','claude',?,0,0)`, junk); err != nil {
		t.Fatal(err)
	}

	out, err := Optimize(db, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "junk repo") {
		t.Errorf("optimize output should report the relabel, got:\n%s", out)
	}

	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM facts WHERE repo = ?`, junk).Scan(&n); err != nil || n != 0 {
		t.Errorf("junk repo should be gone from facts, %d rows left (err %v)", n, err)
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM files WHERE repo = ?`, junk).Scan(&n); err != nil || n != 0 {
		t.Errorf("junk repo should be gone from files, %d rows left (err %v)", n, err)
	}
	var repo string
	if err := db.QueryRow(`SELECT repo FROM facts WHERE id = ?`, keep).Scan(&repo); err != nil || repo != "quasar" {
		t.Errorf("valid repo must be untouched, got %q (err %v)", repo, err)
	}
	var relabeled string
	if err := db.QueryRow(`SELECT repo FROM facts WHERE repo = 'unknown'`).Scan(&relabeled); err != nil {
		t.Errorf("relabeled fact should now be under unknown: %v", err)
	}
}
