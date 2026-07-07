package truth

// oracle repair: batch re-audit of every existing supersession pair with the
// current (upgraded) judge, reopening wrongly-closed facts. Resumable via the
// repair_done table; safe to kill and restart. Fails loudly per ADR-004 style:
// a failed batch is retried once, then skipped WITHOUT marking repair_done so
// a rerun picks it up.

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const repairSchema = `
CREATE TABLE IF NOT EXISTS repair_done(
  old_id INTEGER PRIMARY KEY,
  verdict TEXT NOT NULL,
  ts REAL NOT NULL
);`

const repairPrompt = `You are an independent skeptical auditor of a bi-temporal fact graph. Each PAIR below records that fact OLD was closed as superseded by fact NEW. Many of these closures were made by a weaker judge and may be wrong. Re-decide each pair on its merits.

Verdicts:
- "UPHOLD": NEW is a true replacement of OLD — same topic, same aspect, NEW states the newer/current state (or is an exact duplicate).
- "REOPEN": the closure was wrong — OLD carries information NEW does not (information lost), or the facts are unrelated, or they describe DIFFERENT ASPECTS of the same system. Different aspect != replacement.
- "REOPEN_CONTRADICT": the closure was wrong AND the two facts actively conflict — both claim to be the current state and cannot both be true.

Bias instructions:
- DATES MATTER. A fact may only be replaced by a NEWER fact: if NEW's valid_from date is before OLD's, it cannot supersede it — REOPEN (or REOPEN_CONTRADICT if they conflict).
- Evidence tiers: verified > asserted > reported. Be more skeptical of a low-tier NEW fact closing a verified OLD fact.
- When in doubt whether NEW fully covers OLD's information, prefer REOPEN — a wrongly-open fact is cheap, a wrongly-buried fact is lost.

Return JSON: {"verdicts":[{"old_id":int, "verdict":"UPHOLD"|"REOPEN"|"REOPEN_CONTRADICT"}]} with exactly one entry per pair given.`

type repairPair struct {
	OldID, NewID               int64
	OldStmt, NewStmt           string
	OldKind, NewKind           string
	OldRepo, NewRepo           string
	OldValidFrom, NewValidFrom float64
	OldEvidence, NewEvidence   string
}

func applyRepairSchema(db *sql.DB) error {
	if _, err := db.Exec(repairSchema); err != nil {
		return fmt.Errorf("apply repair schema: %w", err)
	}
	return nil
}

// loadRepairPairs returns all not-yet-audited supersession pairs, oldest first.
func loadRepairPairs(db store.DBQ, maxPairs int) ([]repairPair, error) {
	q := `SELECT f.id, s.id,
	       f.statement, s.statement,
	       f.kind, s.kind,
	       COALESCE(f.repo,''), COALESCE(s.repo,''),
	       f.valid_from, s.valid_from,
	       f.evidence, s.evidence
	FROM facts f JOIN facts s ON s.id = f.superseded_by
	LEFT JOIN repair_done d ON d.old_id = f.id
	WHERE d.old_id IS NULL
	ORDER BY f.id`
	args := []any{}
	if maxPairs > 0 {
		q += " LIMIT ?"
		args = append(args, maxPairs)
	}
	rows, err := db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []repairPair
	for rows.Next() {
		var p repairPair
		if err := rows.Scan(&p.OldID, &p.NewID, &p.OldStmt, &p.NewStmt,
			&p.OldKind, &p.NewKind, &p.OldRepo, &p.NewRepo,
			&p.OldValidFrom, &p.NewValidFrom, &p.OldEvidence, &p.NewEvidence); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// repairBatchUser renders one LLM batch: full context for both sides of each pair.
func repairBatchUser(pairs []repairPair) string {
	var b strings.Builder
	for _, p := range pairs {
		fmt.Fprintf(&b, "PAIR old_id=%d\n", p.OldID)
		fmt.Fprintf(&b, "  OLD (closed as superseded): %s\n    [kind=%s repo=%s valid_from=%s evidence=%s]\n",
			p.OldStmt, p.OldKind, p.OldRepo, store.AsOfDate(p.OldValidFrom), p.OldEvidence)
		fmt.Fprintf(&b, "  NEW (the claimed replacement): %s\n    [kind=%s repo=%s valid_from=%s evidence=%s]\n",
			p.NewStmt, p.NewKind, p.NewRepo, store.AsOfDate(p.NewValidFrom), p.NewEvidence)
	}
	return b.String()
}

var validRepairVerdicts = map[string]bool{
	"UPHOLD": true, "REOPEN": true, "REOPEN_CONTRADICT": true,
}

// judgeRepairBatch asks the model for one verdict per pair. A missing or
// invalid verdict for any pair is an error — no silent defaults.
// ORACLE_LOCAL_JUDGE shadow/active applies here uniformly (localjudge.go):
// the local judge natively speaks the repair vocabulary; judge_shadow rows
// are logged in the canonical vocabulary. db is only for judge_shadow.
func judgeRepairBatch(db DBE, pairs []repairPair) (map[int64]string, error) {
	mode, err := LocalJudgeMode()
	if err != nil {
		return nil, err
	}
	if mode != LocalJudgeOff {
		if err := EnsureJudgeShadow(db); err != nil {
			return nil, err
		}
	}
	sides := func(p repairPair) (string, string) {
		return RenderJudgeSide(p.OldStmt, store.AsOfDate(p.OldValidFrom), p.OldEvidence, p.OldKind),
			RenderJudgeSide(p.NewStmt, store.AsOfDate(p.NewValidFrom), p.NewEvidence, p.NewKind)
	}

	if mode == LocalJudgeActive {
		threshold, err := LocalJudgeMargin()
		if err != nil {
			return nil, err
		}
		now := float64(time.Now().Unix())
		verdicts := map[int64]string{}
		var residual []repairPair
		for _, p := range pairs {
			oldSide, newSide := sides(p)
			v, ok, err := ActiveJudgePair(db, now, p.OldID, p.NewID, threshold, oldSide, newSide)
			if err != nil {
				return nil, err
			}
			if !ok {
				residual = append(residual, p)
				continue
			}
			verdicts[p.OldID] = canonicalToRepair[v]
		}
		if len(residual) > 0 {
			llmV, err := judgeRepairBatchLLM(residual)
			if err != nil {
				return nil, err
			}
			for id, v := range llmV {
				verdicts[id] = v
			}
		}
		return verdicts, nil
	}

	verdicts, err := judgeRepairBatchLLM(pairs)
	if err != nil {
		return nil, err
	}
	if mode == LocalJudgeShadow {
		now := float64(time.Now().Unix())
		for _, p := range pairs {
			oldSide, newSide := sides(p)
			ShadowJudgePair(db, now, p.OldID, p.NewID, repairToCanonical[verdicts[p.OldID]], oldSide, newSide)
		}
	}
	return verdicts, nil
}

// judgeRepairBatchLLM is the original the frontier LLM batch call, untouched.
func judgeRepairBatchLLM(pairs []repairPair) (map[int64]string, error) {
	var out struct {
		Verdicts []struct {
			OldID   int64  `json:"old_id"`
			Verdict string `json:"verdict"`
		} `json:"verdicts"`
	}
	if err := llm.ChatJSON(repairPrompt, repairBatchUser(pairs), 8000, &out); err != nil {
		return nil, err
	}
	want := map[int64]bool{}
	for _, p := range pairs {
		want[p.OldID] = true
	}
	verdicts := map[int64]string{}
	for _, v := range out.Verdicts {
		vd := strings.ToUpper(strings.TrimSpace(v.Verdict))
		if !want[v.OldID] {
			continue // hallucinated id — drop; the missing-pair check below stays loud
		}
		if !validRepairVerdicts[vd] {
			return nil, fmt.Errorf("pair %d: invalid verdict %q", v.OldID, v.Verdict)
		}
		verdicts[v.OldID] = vd
	}
	for id := range want {
		if _, ok := verdicts[id]; !ok {
			return nil, fmt.Errorf("judge returned no verdict for pair old_id=%d", id)
		}
	}
	return verdicts, nil
}

// applyRepairVerdict applies one pair's verdict + the repair_done marker in a
// single transaction.
func applyRepairVerdict(db *sql.DB, p repairPair, verdict string) error {
	if !validRepairVerdicts[verdict] {
		return fmt.Errorf("pair %d: invalid verdict %q", p.OldID, verdict)
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	now := float64(time.Now().Unix())
	if verdict == "REOPEN" || verdict == "REOPEN_CONTRADICT" {
		if _, err := tx.Exec(`UPDATE facts SET superseded_at = NULL, superseded_by = NULL
			WHERE id = ?`, p.OldID); err != nil {
			return err
		}
		if _, err := tx.Exec(`DELETE FROM edges WHERE src = ? AND dst = ? AND type = 'supersedes'`,
			p.NewID, p.OldID); err != nil {
			return err
		}
	}
	if verdict == "REOPEN_CONTRADICT" {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO edges(src, dst, type, recorded_at)
			VALUES(?,?,?,?)`, p.NewID, p.OldID, "contradicts", now); err != nil {
			return err
		}
	}
	if _, err := tx.Exec(`INSERT INTO repair_done(old_id, verdict, ts) VALUES(?,?,?)`,
		p.OldID, verdict, now); err != nil {
		return err
	}
	return tx.Commit()
}

const repairBatchSize = 8

// runRepair drives the audit. sample>0 = dry run: judge the first N pairs and
// print verdicts without writing anything.
func RunRepair(db *sql.DB, workers, sample, maxPairs int) error {
	if err := applyRepairSchema(db); err != nil {
		return err
	}
	limit := maxPairs
	if sample > 0 {
		limit = sample
	}
	pairs, err := loadRepairPairs(db, limit)
	if err != nil {
		return err
	}
	if len(pairs) == 0 {
		fmt.Println("repair: no pending supersession pairs")
		return nil
	}
	fmt.Printf("repair: %d pairs pending (workers=%d sample=%v)\n", len(pairs), workers, sample > 0)

	if sample > 0 {
		for i := 0; i < len(pairs); i += repairBatchSize {
			batch := pairs[i:min(i+repairBatchSize, len(pairs))]
			verdicts, err := judgeRepairBatch(db, batch)
			if err != nil {
				return fmt.Errorf("sample batch at pair %d: %w", batch[0].OldID, err)
			}
			for _, p := range batch {
				fmt.Printf("%-17s old=%d %q  <-  new=%d %q\n",
					verdicts[p.OldID], p.OldID, store.Truncate(p.OldStmt, 90), p.NewID, store.Truncate(p.NewStmt, 90))
			}
		}
		return nil
	}

	if workers < 1 {
		return fmt.Errorf("workers must be >= 1, got %d", workers)
	}
	// Partition pairs by old_id % workers so a restart with the same worker
	// count reproduces the same partitions.
	parts := make([][]repairPair, workers)
	for _, p := range pairs {
		w := int(p.OldID % int64(workers))
		parts[w] = append(parts[w], p)
	}

	var done, reopened, contradicted, skipped atomic.Int64
	total := int64(len(pairs))
	start := time.Now()
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for w := 0; w < workers; w++ {
		mine := parts[w]
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < len(mine); i += repairBatchSize {
				batch := mine[i:min(i+repairBatchSize, len(mine))]
				verdicts, err := judgeRepairBatch(db, batch)
				if err != nil {
					fmt.Printf("repair: batch at old_id=%d failed (%v), retrying once\n", batch[0].OldID, err)
					verdicts, err = judgeRepairBatch(db, batch)
				}
				if err != nil {
					// skip WITHOUT repair_done so a rerun picks these up
					fmt.Printf("repair: batch at old_id=%d failed twice, skipping %d pairs: %v\n",
						batch[0].OldID, len(batch), err)
					skipped.Add(int64(len(batch)))
					continue
				}
				for _, p := range batch {
					v := verdicts[p.OldID]
					if err := applyRepairVerdict(db, p, v); err != nil {
						errCh <- fmt.Errorf("apply pair %d: %w", p.OldID, err)
						return
					}
					switch v {
					case "REOPEN":
						reopened.Add(1)
					case "REOPEN_CONTRADICT":
						reopened.Add(1)
						contradicted.Add(1)
					}
					if n := done.Add(1); n%200 == 0 {
						fmt.Printf("repair: %d/%d done (%d reopened, %d contradicts, %d skipped) %.0fs\n",
							n, total, reopened.Load(), contradicted.Load(), skipped.Load(),
							time.Since(start).Seconds())
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		return err // DB write failures are fatal — never swallow
	}
	fmt.Printf("repair: finished %d/%d pairs — %d upheld, %d reopened (%d with contradicts edge), %d skipped\n",
		done.Load(), total, done.Load()-reopened.Load(), reopened.Load(), contradicted.Load(), skipped.Load())
	if s := skipped.Load(); s > 0 {
		return fmt.Errorf("repair: %d pairs skipped after retry — rerun 'oracle repair' to pick them up", s)
	}
	return nil
}

// repairStatus is a tiny helper for tests/CLI sanity: verdict counts so far.
func repairStatus(db store.DBQ) (map[string]int, error) {
	rows, err := db.Query(`SELECT verdict, COUNT(*) FROM repair_done GROUP BY verdict`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var v string
		var c int
		if err := rows.Scan(&v, &c); err != nil {
			return nil, err
		}
		out[v] = c
	}
	return out, rows.Err()
}
