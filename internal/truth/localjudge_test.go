package truth

// Tests for localjudge.go.

import (
	"database/sql"
	"errors"
	"math"
	"oracle/internal/store"
	"testing"
)

func TestLocalVerdictMapping(t *testing.T) {
	cases := []struct {
		probs   [3]float64
		verdict string
		margin  float64
	}{
		// index 0 = UPHOLD ("supersession correct") -> supersedes
		{[3]float64{0.90, 0.06, 0.04}, "supersedes", 0.84},
		// index 1 = REOPEN ("no relation strong enough") -> nothing
		{[3]float64{0.10, 0.70, 0.20}, "nothing", 0.50},
		// index 2 = REOPEN_CONTRADICT ("they conflict") -> contradicts
		{[3]float64{0.05, 0.15, 0.80}, "contradicts", 0.65},
		// margin = top minus SECOND, not top minus sum-of-rest
		{[3]float64{0.50, 0.45, 0.05}, "supersedes", 0.05},
	}
	for _, c := range cases {
		v, m := localVerdictFromProbs(c.probs)
		if v != c.verdict {
			t.Errorf("probs %v: verdict %q, want %q", c.probs, v, c.verdict)
		}
		if math.Abs(m-c.margin) > 1e-9 {
			t.Errorf("probs %v: margin %.4f, want %.4f", c.probs, m, c.margin)
		}
	}
	// repair <-> canonical maps must be exact inverses over the full vocab.
	for r, cv := range repairToCanonical {
		if canonicalToRepair[cv] != r {
			t.Errorf("mapping not inverse: %s -> %s -> %s", r, cv, canonicalToRepair[cv])
		}
	}
	if len(repairToCanonical) != 3 || len(canonicalToRepair) != 3 {
		t.Errorf("mapping must cover exactly the 3 verdicts")
	}
}

func shadowRow(t *testing.T, db *sql.DB) (oldID int64, newID sql.NullInt64, llm, local string, margin float64, n int) {
	t.Helper()
	if err := db.QueryRow(`SELECT COUNT(*) FROM judge_shadow`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		return
	}
	if err := db.QueryRow(`SELECT old_id, new_id, llm_verdict, local_verdict, local_margin
		FROM judge_shadow LIMIT 1`).Scan(&oldID, &newID, &llm, &local, &margin); err != nil {
		t.Fatal(err)
	}
	return
}

func TestShadowLoggingWithStub(t *testing.T) {
	db := store.TestDB(t)
	if err := EnsureJudgeShadow(db); err != nil {
		t.Fatal(err)
	}

	// Fake local judge: confident UPHOLD.
	orig := judgeLocalImpl
	defer func() { judgeLocalImpl = orig }()
	judgeLocalImpl = func(oldSide, newSide string) ([3]float64, error) {
		return [3]float64{0.95, 0.03, 0.02}, nil
	}

	ShadowJudgePair(db, 1000, 42, 43, "supersedes", "[old side]", "[new side]")
	oldID, newID, llm, local, margin, n := shadowRow(t, db)
	if n != 1 {
		t.Fatalf("want 1 judge_shadow row, got %d", n)
	}
	if oldID != 42 || !newID.Valid || newID.Int64 != 43 {
		t.Errorf("ids: old=%d new=%v", oldID, newID)
	}
	if llm != "supersedes" || local != "supersedes" {
		t.Errorf("verdicts: llm=%q local=%q", llm, local)
	}
	if math.Abs(margin-0.92) > 1e-9 {
		t.Errorf("margin %.4f, want 0.92", margin)
	}

	// NULL new_id for ingest-path pairs (newID <= 0).
	if _, err := db.Exec(`DELETE FROM judge_shadow`); err != nil {
		t.Fatal(err)
	}
	ShadowJudgePair(db, 1000, 7, 0, "nothing", "[old]", "[new]")
	_, newID, _, _, _, n = shadowRow(t, db)
	if n != 1 || newID.Valid {
		t.Errorf("ingest pair: rows=%d new_id=%v, want 1 row with NULL new_id", n, newID)
	}

	// Shadow NEVER breaks the write path: a failing local judge logs nothing
	// and returns normally.
	if _, err := db.Exec(`DELETE FROM judge_shadow`); err != nil {
		t.Fatal(err)
	}
	judgeLocalImpl = func(oldSide, newSide string) ([3]float64, error) {
		return [3]float64{}, errors.New("model not loaded")
	}
	ShadowJudgePair(db, 1000, 8, 9, "contradicts", "[old]", "[new]")
	_, _, _, _, _, n = shadowRow(t, db)
	if n != 0 {
		t.Errorf("failed local judge must log nothing, got %d rows", n)
	}
}

func TestActiveMarginGate(t *testing.T) {
	db := store.TestDB(t)
	if err := EnsureJudgeShadow(db); err != nil {
		t.Fatal(err)
	}
	orig := judgeLocalImpl
	defer func() { judgeLocalImpl = orig }()

	// Above threshold: local verdict taken, logged with llm_verdict=''.
	judgeLocalImpl = func(oldSide, newSide string) ([3]float64, error) {
		return [3]float64{0.02, 0.03, 0.95}, nil // contradicts, margin 0.92
	}
	v, ok, err := ActiveJudgePair(db, 1000, 5, 6, 0.85, "[old]", "[new]")
	if err != nil || !ok || v != "contradicts" {
		t.Fatalf("above threshold: v=%q ok=%v err=%v", v, ok, err)
	}
	_, _, llm, local, _, n := shadowRow(t, db)
	if n != 1 || llm != "" || local != "contradicts" {
		t.Errorf("active log: rows=%d llm=%q local=%q", n, llm, local)
	}

	// Below threshold: not taken, not logged.
	if _, err := db.Exec(`DELETE FROM judge_shadow`); err != nil {
		t.Fatal(err)
	}
	judgeLocalImpl = func(oldSide, newSide string) ([3]float64, error) {
		return [3]float64{0.5, 0.4, 0.1}, nil // margin 0.10
	}
	_, ok, err = ActiveJudgePair(db, 1000, 5, 6, 0.85, "[old]", "[new]")
	if err != nil || ok {
		t.Fatalf("below threshold: ok=%v err=%v", ok, err)
	}
	if _, _, _, _, _, n := shadowRow(t, db); n != 0 {
		t.Errorf("below-threshold pair must not be logged, got %d rows", n)
	}

	// Active mode fails loudly on local-judge error.
	judgeLocalImpl = func(oldSide, newSide string) ([3]float64, error) {
		return [3]float64{}, errors.New("boom")
	}
	if _, _, err := ActiveJudgePair(db, 1000, 5, 6, 0.85, "[old]", "[new]"); err == nil {
		t.Error("active mode must propagate local-judge errors")
	}
}

func TestLocalJudgeEnvConfig(t *testing.T) {
	t.Setenv("ORACLE_LOCAL_JUDGE", "")
	if m, err := LocalJudgeMode(); err != nil || m != LocalJudgeOff {
		t.Errorf("default mode: %q %v", m, err)
	}
	t.Setenv("ORACLE_LOCAL_JUDGE", "shadow")
	if m, _ := LocalJudgeMode(); m != LocalJudgeShadow {
		t.Errorf("shadow mode: %q", m)
	}
	t.Setenv("ORACLE_LOCAL_JUDGE", "bogus")
	if _, err := LocalJudgeMode(); err == nil {
		t.Error("invalid mode must be a loud error, not silent off")
	}

	t.Setenv("ORACLE_JUDGE_MARGIN", "")
	if v, err := LocalJudgeMargin(); err != nil || v != 0.85 {
		t.Errorf("default margin: %v %v", v, err)
	}
	t.Setenv("ORACLE_JUDGE_MARGIN", "0.9")
	if v, _ := LocalJudgeMargin(); v != 0.9 {
		t.Errorf("margin: %v", v)
	}
	t.Setenv("ORACLE_JUDGE_MARGIN", "nope")
	if _, err := LocalJudgeMargin(); err == nil {
		t.Error("invalid margin must error")
	}
}
