package truth

// oracle referee: resolve live contradiction pairs (edges type='contradicts'
// where both facts are live) with a frontier-LLM verdict per pair.
// Research rule (docs/truth_design.md): prefer keep-with-provenance; force
// SUPERSEDE only on clear temporal replacement. Never delete a fact.

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"strings"
	"sync"
	"time"
)

const refereePrompt = `You referee a bi-temporal fact graph. Two facts below were flagged as CONTRADICTING each other and both are still marked live. Decide ONE verdict:

- "SUPERSEDE_NEWER": the world changed — the fact with the LATER valid_from date cleanly replaces the older one (same claim, newer state). Only pick this on CLEAR temporal replacement of the same claim about the same thing.
- "DIFFERENT_SCOPE": both are true — they describe different aspects, components, time-scopes, environments, or configurations. The contradiction flag was wrong.
- "UNRESOLVED": genuinely conflicting current claims, or not enough information to be sure. When in doubt, pick this — keeping both visible with provenance beats forcing a wrong winner.

Weigh evidence tiers (verified > asserted > reported), corroborations, and dates. Do NOT supersede an older VERIFIED fact with a newer merely-REPORTED one unless the replacement is unmistakable.

Return JSON: {"verdict":"SUPERSEDE_NEWER|DIFFERENT_SCOPE|UNRESOLVED","reason":"one sentence"}`

type refFact struct {
	ID             int64
	Statement      string
	Kind           string
	Repo           string
	ValidFrom      float64
	RecordedAt     float64
	Evidence       string
	Confidence     float64
	Corroborations int
}

type conflictPair struct {
	A, B refFact // A = edge src, B = edge dst
}

// newerOlder orders the pair by world time (valid_from; recorded_at breaks ties).
func (p conflictPair) newerOlder() (refFact, refFact) {
	if p.A.ValidFrom > p.B.ValidFrom ||
		(p.A.ValidFrom == p.B.ValidFrom && p.A.RecordedAt >= p.B.RecordedAt) {
		return p.A, p.B
	}
	return p.B, p.A
}

// liveConflicts returns contradicts edges where BOTH ends are live.
// skipDone excludes pairs already refereed (resume).
func liveConflicts(db *sql.DB, skipDone bool) ([]conflictPair, error) {
	q := `SELECT a.id, a.statement, a.kind, COALESCE(a.repo,''), a.valid_from, a.recorded_at,
	             a.evidence, a.confidence, a.corroborations,
	             b.id, b.statement, b.kind, COALESCE(b.repo,''), b.valid_from, b.recorded_at,
	             b.evidence, b.confidence, b.corroborations
	      FROM edges e
	      JOIN facts a ON a.id = e.src AND a.superseded_at IS NULL
	      JOIN facts b ON b.id = e.dst AND b.superseded_at IS NULL
	      WHERE e.type = 'contradicts'`
	if skipDone {
		q += ` AND NOT EXISTS (SELECT 1 FROM referee_done d
		        WHERE (d.src = e.src AND d.dst = e.dst) OR (d.src = e.dst AND d.dst = e.src))`
	}
	q += ` ORDER BY e.src, e.dst`
	rows, err := db.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []conflictPair
	for rows.Next() {
		var p conflictPair
		if err := rows.Scan(
			&p.A.ID, &p.A.Statement, &p.A.Kind, &p.A.Repo, &p.A.ValidFrom, &p.A.RecordedAt,
			&p.A.Evidence, &p.A.Confidence, &p.A.Corroborations,
			&p.B.ID, &p.B.Statement, &p.B.Kind, &p.B.Repo, &p.B.ValidFrom, &p.B.RecordedAt,
			&p.B.Evidence, &p.B.Confidence, &p.B.Corroborations); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// listConflicts prints live contradiction pairs. Plain output, no LLM.
func ListConflicts(db *sql.DB) error {
	pairs, err := liveConflicts(db, false)
	if err != nil {
		return err
	}
	for _, p := range pairs {
		fmt.Printf("%d <> %d  [%s|%s vs %s|%s]\n  A: %s\n  B: %s\n",
			p.A.ID, p.B.ID,
			store.LocalDate(p.A.ValidFrom), p.A.Evidence, store.LocalDate(p.B.ValidFrom), p.B.Evidence,
			store.Truncate(p.A.Statement, 140), store.Truncate(p.B.Statement, 140))
	}
	fmt.Printf("%d live contradiction pair(s)\n", len(pairs))
	return nil
}

func refFactBlock(label string, f refFact) string {
	return fmt.Sprintf("%s (id %d):\n  statement: %s\n  kind: %s | repo: %s\n  valid_from: %s | recorded: %s\n  evidence: %s | confidence: %.2f | corroborations: %d\n",
		label, f.ID, f.Statement, f.Kind, f.Repo,
		store.LocalDate(f.ValidFrom), store.LocalDate(f.RecordedAt), f.Evidence, f.Confidence, f.Corroborations)
}

var refereeVerdicts = map[string]bool{
	"SUPERSEDE_NEWER": true, "DIFFERENT_SCOPE": true, "UNRESOLVED": true,
}

func judgeConflict(p conflictPair) (verdict, reason string, err error) {
	user := refFactBlock("FACT A", p.A) + "\n" + refFactBlock("FACT B", p.B)
	var out struct {
		Verdict string `json:"verdict"`
		Reason  string `json:"reason"`
	}
	if err := llm.ChatJSON(refereePrompt, user, 4000, &out); err != nil {
		return "", "", err
	}
	out.Verdict = strings.ToUpper(strings.TrimSpace(out.Verdict))
	if !refereeVerdicts[out.Verdict] {
		return "", "", fmt.Errorf("pair %d<>%d: invalid verdict %q", p.A.ID, p.B.ID, out.Verdict)
	}
	return out.Verdict, out.Reason, nil
}

// applyVerdict applies one referee verdict in its own transaction.
// Never deletes a fact; only stamps supersession / removes the contradicts edge.
func applyVerdict(db *sql.DB, p conflictPair, verdict string) error {
	if !refereeVerdicts[verdict] {
		return fmt.Errorf("unknown verdict %q", verdict)
	}
	now := float64(time.Now().Unix())
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	switch verdict {
	case "SUPERSEDE_NEWER":
		newer, older := p.newerOlder()
		if _, err := tx.Exec(`UPDATE facts SET superseded_at = ?, superseded_by = ?
			WHERE id = ? AND superseded_at IS NULL`, now, newer.ID, older.ID); err != nil {
			return err
		}
		if _, err := tx.Exec(`DELETE FROM edges WHERE type = 'contradicts'
			AND ((src = ? AND dst = ?) OR (src = ? AND dst = ?))`,
			p.A.ID, p.B.ID, p.B.ID, p.A.ID); err != nil {
			return err
		}
		if _, err := tx.Exec(`INSERT OR IGNORE INTO edges(src, dst, type, recorded_at)
			VALUES(?,?,?,?)`, newer.ID, older.ID, "supersedes", now); err != nil {
			return err
		}
	case "DIFFERENT_SCOPE":
		if _, err := tx.Exec(`DELETE FROM edges WHERE type = 'contradicts'
			AND ((src = ? AND dst = ?) OR (src = ? AND dst = ?))`,
			p.A.ID, p.B.ID, p.B.ID, p.A.ID); err != nil {
			return err
		}
	case "UNRESOLVED":
		// keep the edge; the pair stays visible in `oracle conflicts`
	}
	if _, err := tx.Exec(`INSERT OR REPLACE INTO referee_done(src, dst, verdict, decided_at)
		VALUES(?,?,?,?)`, p.A.ID, p.B.ID, verdict, now); err != nil {
		return err
	}
	return tx.Commit()
}

// referee judges every unresolved live contradiction pair. LLM calls fan out
// across workers; verdict application stays serial (single sqlite writer).
// Returns verdict -> count.
func Referee(db *sql.DB, workers int, dryRun bool) (map[string]int, error) {
	if workers < 1 {
		workers = 1
	}
	pairs, err := liveConflicts(db, true)
	if err != nil {
		return nil, err
	}
	fmt.Printf("refereeing %d pair(s), workers=%d dry-run=%v\n", len(pairs), workers, dryRun)
	type result struct {
		pair    conflictPair
		verdict string
		reason  string
		err     error
	}
	jobs := make(chan conflictPair)
	results := make(chan result)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range jobs {
				v, r, err := judgeConflict(p)
				results <- result{p, v, r, err}
			}
		}()
	}
	go func() {
		for _, p := range pairs {
			jobs <- p
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()
	counts := map[string]int{}
	var firstErr error
	done := 0
	for r := range results {
		done++
		if r.err != nil {
			counts["ERROR"]++
			fmt.Printf("[%d/%d] %d<>%d ERROR: %v\n", done, len(pairs), r.pair.A.ID, r.pair.B.ID, r.err)
			if firstErr == nil {
				firstErr = r.err
			}
			continue
		}
		counts[r.verdict]++
		fmt.Printf("[%d/%d] %d<>%d %s — %s\n", done, len(pairs), r.pair.A.ID, r.pair.B.ID,
			r.verdict, store.Truncate(r.reason, 160))
		if dryRun {
			continue
		}
		if err := applyVerdict(db, r.pair, r.verdict); err != nil {
			return counts, fmt.Errorf("apply %d<>%d: %w", r.pair.A.ID, r.pair.B.ID, err)
		}
	}
	if firstErr != nil {
		return counts, fmt.Errorf("%d pair(s) failed, first: %w", counts["ERROR"], firstErr)
	}
	return counts, nil
}
