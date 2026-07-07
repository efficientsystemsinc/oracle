package truth

// localjudge.go — margin-gated local 110M judge on the supersession write
// paths (ingest judgeSupersede, sweep, repair). Modes via ORACLE_LOCAL_JUDGE:
//
//	off    (default) — the frontier LLM only, exactly as before.
//	shadow — the frontier LLM decides; the local judge also runs per pair and its
//	         agreement is logged to judge_shadow. ZERO behavior change:
//	         local-judge errors are warned, never propagated.
//	active — pairs where the local judge's margin (top prob − second prob)
//	         >= ORACLE_JUDGE_MARGIN (default 0.85) take the local verdict
//	         WITHOUT a frontier-LLM call (logged with llm_verdict=''); the rest
//	         batch to the frontier LLM exactly as today. Local errors are fatal here
//	         (fail loud — active mode is on the decision path).
//
// judgeLocal(oldSide, newSide) is provided by the local-model integration
// (stubbed in infer_stub2.go on this branch); callers go through
// judgeLocalImpl so tests can substitute fake probabilities.

import (
	"database/sql"
	"fmt"
	"oracle/internal/infer"
	"os"
	"strconv"
)

// ---------------------------------------------------------------------------
// THE verdict mapping — single source of truth.
//
// The local model was trained on the REPAIR vocabulary, where the question is
// "was this supersession correct?":
//
//	index 0  UPHOLD            = "the supersession is correct"   -> supersedes
//	index 1  REOPEN            = "no relation strong enough"     -> nothing
//	index 2  REOPEN_CONTRADICT = "the two facts conflict"        -> contradicts
//
// So in the ingest/sweep vocabulary a REPLACEMENT verdict ("supersedes") is
// the UPHOLD class, "nothing" is REOPEN, and "contradicts" is
// REOPEN_CONTRADICT. judge_shadow stores ONLY the canonical ingest vocabulary
// (supersedes|nothing|contradicts) for both llm_verdict and local_verdict so
// agreement aggregates uniformly across all three paths.
// ---------------------------------------------------------------------------

// localVerdictByIndex maps a judgeLocal probability index -> canonical verdict.
var localVerdictByIndex = [3]string{"supersedes", "nothing", "contradicts"}

// repairToCanonical maps repair-vocabulary verdicts -> canonical verdicts
// (used when logging repair-path LLM verdicts into judge_shadow).
var repairToCanonical = map[string]string{
	"UPHOLD":            "supersedes",
	"REOPEN":            "nothing",
	"REOPEN_CONTRADICT": "contradicts",
}

// canonicalToRepair is the inverse, for the repair path consuming a local verdict.
var canonicalToRepair = map[string]string{
	"supersedes":  "UPHOLD",
	"nothing":     "REOPEN",
	"contradicts": "REOPEN_CONTRADICT",
}

// localVerdictFromProbs picks the argmax verdict and the margin
// (top probability minus second probability).
func localVerdictFromProbs(p [3]float64) (verdict string, margin float64) {
	best := 0
	for i := 1; i < 3; i++ {
		if p[i] > p[best] {
			best = i
		}
	}
	second := -1.0
	for i := 0; i < 3; i++ {
		if i != best && p[i] > second {
			second = p[i]
		}
	}
	return localVerdictByIndex[best], p[best] - second
}

// judgeLocalImpl is the seam tests use to fake probabilities; production code
// must call this, never judgeLocal directly.
var judgeLocalImpl = infer.JudgeLocal

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const (
	LocalJudgeOff    = "off"
	LocalJudgeShadow = "shadow"
	LocalJudgeActive = "active"

	defaultJudgeMargin = 0.85
)

// localJudgeMode reads ORACLE_LOCAL_JUDGE. Unset/empty = off. Anything else
// invalid is a loud error, not a silent off.
func LocalJudgeMode() (string, error) {
	switch v := os.Getenv("ORACLE_LOCAL_JUDGE"); v {
	case "", LocalJudgeOff:
		return LocalJudgeOff, nil
	case LocalJudgeShadow, LocalJudgeActive:
		return v, nil
	default:
		return "", fmt.Errorf("ORACLE_LOCAL_JUDGE=%q: want off|shadow|active", v)
	}
}

// localJudgeMargin reads ORACLE_JUDGE_MARGIN (default 0.85).
func LocalJudgeMargin() (float64, error) {
	v := os.Getenv("ORACLE_JUDGE_MARGIN")
	if v == "" {
		return defaultJudgeMargin, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f <= 0 || f > 1 {
		return 0, fmt.Errorf("ORACLE_JUDGE_MARGIN=%q: want a float in (0,1]", v)
	}
	return f, nil
}

// ---------------------------------------------------------------------------
// judge_shadow
// ---------------------------------------------------------------------------

const judgeShadowSchema = `
CREATE TABLE IF NOT EXISTS judge_shadow(
  ts REAL NOT NULL,
  old_id INTEGER NOT NULL,
  new_id INTEGER,
  llm_verdict TEXT NOT NULL,
  local_verdict TEXT NOT NULL,
  local_margin REAL NOT NULL
);`

type DBE interface {
	Exec(query string, args ...any) (sql.Result, error)
}

func EnsureJudgeShadow(db DBE) error {
	if _, err := db.Exec(judgeShadowSchema); err != nil {
		return fmt.Errorf("apply judge_shadow schema: %w", err)
	}
	return nil
}

// logJudgeShadow inserts one comparison row. newID <= 0 means "no new-fact id
// yet" (ingest judges before the insert) and is stored as NULL.
func logJudgeShadow(db DBE, ts float64, oldID, newID int64, llmVerdict, localVerdict string, margin float64) error {
	var nid any
	if newID > 0 {
		nid = newID
	}
	_, err := db.Exec(`INSERT INTO judge_shadow(ts, old_id, new_id, llm_verdict, local_verdict, local_margin)
		VALUES(?,?,?,?,?,?)`, ts, oldID, nid, llmVerdict, localVerdict, margin)
	return err
}

// renderJudgeSide renders one side of a pair the way judgeUserPrompt renders
// facts, plus kind — the exact string contract judgeLocal was trained on.
func RenderJudgeSide(stmt, date, evidence, kind string) string {
	return fmt.Sprintf("[valid_from %s][evidence %s][kind %s] %s", date, evidence, kind, stmt)
}

// shadowJudgePair runs the local judge alongside a frontier-LLM verdict and logs
// agreement. NEVER fails the write path: any error is a stderr warning
// (shadow mode's contract is zero behavior change).
func ShadowJudgePair(db DBE, ts float64, oldID, newID int64, llmVerdict, oldSide, newSide string) {
	probs, err := judgeLocalImpl(oldSide, newSide)
	if err != nil {
		fmt.Fprintf(os.Stderr, "local-judge shadow: pair old=%d: %v\n", oldID, err)
		return
	}
	verdict, margin := localVerdictFromProbs(probs)
	if err := logJudgeShadow(db, ts, oldID, newID, llmVerdict, verdict, margin); err != nil {
		fmt.Fprintf(os.Stderr, "local-judge shadow: log pair old=%d: %v\n", oldID, err)
	}
}

// activeJudgePair runs the local judge in active mode. If the margin clears
// the threshold, the local verdict is TAKEN (returned ok=true) and logged with
// llm_verdict=”. Below threshold: ok=false, nothing logged, the pair goes to
// the frontier LLM. Errors are fatal — active mode is on the decision path.
func ActiveJudgePair(db DBE, ts float64, oldID, newID int64, threshold float64, oldSide, newSide string) (verdict string, ok bool, err error) {
	probs, err := judgeLocalImpl(oldSide, newSide)
	if err != nil {
		return "", false, fmt.Errorf("local judge pair old=%d: %w", oldID, err)
	}
	verdict, margin := localVerdictFromProbs(probs)
	if margin < threshold {
		return "", false, nil
	}
	if err := logJudgeShadow(db, ts, oldID, newID, "", verdict, margin); err != nil {
		return "", false, err
	}
	return verdict, true, nil
}
