package search

// Tests for graph.go.

import (
	"math"
	"oracle/internal/ingest"
	"oracle/internal/kb"
	"oracle/internal/store"
	"strings"
	"testing"
)

func TestMassNow(t *testing.T) {
	// fresh fact: full mass plus floor
	if got := massNow("fact", 1.0, 1000, 1000); math.Abs(got-1.05) > 1e-9 {
		t.Errorf("age 0: got %v want 1.05", got)
	}
	// one half-life for status (7d) decays to ~half
	got := massNow("status", 1.0, 0, 7*86400)
	if math.Abs(got-(massEps+0.5)) > 0.01 {
		t.Errorf("one half-life: got %v want ~%v", got, massEps+0.5)
	}
	// unknown kind falls back to 60d half-life
	got = massNow("mystery", 1.0, 0, 60*86400)
	if math.Abs(got-(massEps+0.5)) > 0.01 {
		t.Errorf("default half-life: got %v want ~%v", got, massEps+0.5)
	}
	// preferences outlive status by design
	if massNow("preference", 1.0, 0, 30*86400) <= massNow("status", 1.0, 0, 30*86400) {
		t.Error("preference must decay slower than status")
	}
	// clock skew: valid_from in the future must not inflate mass
	if got := massNow("fact", 1.0, 2000, 1000); got > 1.05+1e-9 {
		t.Errorf("future valid_from must clamp age to 0, got %v", got)
	}
}

func TestFTSQueryDropsStopwords(t *testing.T) {
	q := ftsQuery("which user do you ssh into the meadow boxes as")
	for _, want := range []string{`"user"`, `"ssh"`, `"meadow"`, `"boxes"`} {
		if !strings.Contains(q, want) {
			t.Errorf("missing %s in %s", want, q)
		}
	}
	for _, junk := range []string{`"which"`, `"do"`, `"you"`, `"into"`, `"the"`, `"as"`} {
		if strings.Contains(q, junk) {
			t.Errorf("stopword %s survived in %s", junk, q)
		}
	}
}

// A rare term late in a long question must survive the 12-token cap.
func TestFTSQueryRareTermSurvivesCap(t *testing.T) {
	q := ftsQuery("so what is it that we should have been doing about the thing where they could not get one out for pgbouncer")
	if !strings.Contains(q, `"pgbouncer"`) {
		t.Errorf("rare term lost: %s", q)
	}
}

func TestFTSQueryAllStopwordsFallsBack(t *testing.T) {
	q := ftsQuery("what is the")
	if q == `""` || q == "" {
		t.Errorf("all-stopword query should fall back to raw tokens, got %q", q)
	}
}

// A stale fact about atlas01 recorded under repo "quasar" must surface as a
// supersede candidate for a new atlas01 fact arriving under repo "unknown".
func TestSupersedeCandidatesCrossRepoViaEntity(t *testing.T) {
	db := store.TestDB(t)
	old := store.InsertFact(t, db, "atlas01 is PROD, never run experiments on it", "status", "quasar", 100)
	atlas01 := kb.InsertEntity(t, db, "atlas01")
	if _, err := db.Exec(`INSERT INTO fact_entities(fact_id, entity_id) VALUES(?,?)`, old, atlas01); err != nil {
		t.Fatal(err)
	}
	newFact := ingest.Fact{Statement: "atlas01 is decommissioned, free for experiments",
		Kind: "status", Entities: []string{"atlas01"}}

	got, err := supersedeCandidates(db, newFact, "unknown")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != old {
		t.Errorf("expected cross-repo candidate %d, got %+v", old, got)
	}

	// no entity overlap + different repo -> not a candidate
	noEnt := ingest.Fact{Statement: "atlas01 is decommissioned, free for experiments", Kind: "status"}
	got, err = supersedeCandidates(db, noEnt, "unknown")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("expected no candidates without repo or entity overlap, got %+v", got)
	}

	// same repo still works with no entities at all
	got, err = supersedeCandidates(db, noEnt, "quasar")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Errorf("expected same-repo candidate, got %+v", got)
	}

	// different kind never matches
	wrongKind := ingest.Fact{Statement: "atlas01 is decommissioned", Kind: "decision", Entities: []string{"atlas01"}}
	got, err = supersedeCandidates(db, wrongKind, "quasar")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("kind mismatch should exclude candidates, got %+v", got)
	}

	// entity scoping is exact/alias only: "kml" must NOT prefix-resolve to
	// "atlas01" and widen the candidate set (that fallback is for human input)
	prefix := ingest.Fact{Statement: "atlas01 is decommissioned, free for experiments",
		Kind: "status", Entities: []string{"kml"}}
	got, err = supersedeCandidates(db, prefix, "unknown")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("prefix resolution must not scope candidates, got %+v", got)
	}
}

func TestFTSQueryEmptyAndDedup(t *testing.T) {
	if q := ftsQuery("!!! ?"); q != `""` {
		t.Errorf("no tokens should give empty match, got %q", q)
	}
	q := ftsQuery("quasar quasar QUASAR eval")
	if strings.Count(q, `"quasar"`)+strings.Count(q, `"QUASAR"`) != 1 {
		t.Errorf("expected case-insensitive dedup, got %s", q)
	}
}

// Supersede candidates must carry valid_from + evidence through to the judge
// prompt so the date/tier rules have data to act on.
func TestSupersedeCandidatesCarryDateAndEvidence(t *testing.T) {
	db := store.TestDB(t)
	old := store.InsertFact(t, db, "api runs on box-a in us-east4", "fact", "quasar", 1700000000)
	if _, err := db.Exec(`UPDATE facts SET evidence='verified' WHERE id=?`, old); err != nil {
		t.Fatal(err)
	}
	newFact := ingest.Fact{Statement: "api runs on box-b in us-east4", Kind: "fact", Evidence: "asserted"}
	got, err := supersedeCandidates(db, newFact, "quasar")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 candidate, got %+v", got)
	}
	if got[0].ValidFrom != 1700000000 || got[0].Evidence != "verified" {
		t.Errorf("candidate missing date/evidence: %+v", got[0])
	}

	prompt := ingest.JudgeUserPrompt([]ingest.Fact{newFact}, map[int][]ingest.OldFact{0: got}, "2026-07-05")
	for _, want := range []string{
		"NEW 0 [valid_from 2026-07-05, evidence asserted]",
		"[valid_from 2023-11-14, evidence verified]",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("judge prompt missing %q:\n%s", want, prompt)
		}
	}
}
