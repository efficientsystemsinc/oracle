package truth

// judgeaudit: sample historical supersession pairs and have the frontier LLM re-verify
// each verdict as an independent skeptic. Measures the false-supersession rate
// (pairs where replacing OLD with NEW lost information / wasn't a true
// replacement). Read-only; fails loudly on any LLM or DB error.

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"sync"
)

const auditPrompt = `You are an independent skeptical auditor of a fact-graph's supersession decisions.
You are given one PAIR: an OLD fact that the system marked as superseded (replaced/outdated) by a NEW fact.
Decide whether that replacement was CORRECT: the NEW fact genuinely restates or updates the SAME claim about the SAME subject, so dropping OLD loses nothing still true and distinct.
It is a FALSE supersession if OLD covers a DIFFERENT aspect, entity, number, or condition that NEW does not carry — i.e. information was lost — or if the two are merely related but not a true replacement.
Each fact carries [valid_from YYYY-MM-DD, evidence verified|asserted|reported]. Be extra skeptical when NEW is older-dated than OLD, or when an 'asserted' NEW replaced a 'verified' OLD without explicitly reporting the change.
Return JSON: {"correct": bool, "information_lost": bool, "reason": "one terse sentence"}`

type auditPair struct {
	OldID, NewID     int64
	OldStmt, NewStmt string
	OldVF, NewVF     float64
	OldEv, NewEv     string
}

type auditVerdict struct {
	Correct  bool   `json:"correct"`
	InfoLost bool   `json:"information_lost"`
	Reason   string `json:"reason"`
}

func sampleSupersessions(db *sql.DB, n int) ([]auditPair, error) {
	rows, err := db.Query(`SELECT f.id, f.statement, f.valid_from, f.evidence,
			s.id, s.statement, s.valid_from, s.evidence
		FROM facts f JOIN facts s ON s.id = f.superseded_by
		ORDER BY RANDOM() LIMIT ?`, n)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []auditPair
	for rows.Next() {
		var p auditPair
		if err := rows.Scan(&p.OldID, &p.OldStmt, &p.OldVF, &p.OldEv,
			&p.NewID, &p.NewStmt, &p.NewVF, &p.NewEv); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func auditOne(p auditPair) (auditVerdict, error) {
	user := fmt.Sprintf("OLD (superseded) [valid_from %s, evidence %s]: %s\nNEW (replacement) [valid_from %s, evidence %s]: %s",
		store.AsOfDate(p.OldVF), p.OldEv, p.OldStmt, store.AsOfDate(p.NewVF), p.NewEv, p.NewStmt)
	var v auditVerdict
	if err := llm.ChatJSON(auditPrompt, user, 2000, &v); err != nil {
		return v, fmt.Errorf("pair old=%d new=%d: %w", p.OldID, p.NewID, err)
	}
	return v, nil
}

type auditResult struct {
	Pair    auditPair
	Verdict auditVerdict
}

// judgeAudit runs the audit and prints per-pair verdicts + the overall
// false-supersession rate. Returns the results for further use.
func JudgeAudit(db *sql.DB, sample, workers int) ([]auditResult, error) {
	pairs, err := sampleSupersessions(db, sample)
	if err != nil {
		return nil, err
	}
	if len(pairs) == 0 {
		return nil, fmt.Errorf("no supersession pairs in db")
	}
	results := make([]auditResult, len(pairs))
	errs := make([]error, len(pairs))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for i, p := range pairs {
		wg.Add(1)
		go func(i int, p auditPair) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			v, err := auditOne(p)
			if err != nil {
				errs[i] = err
				return
			}
			results[i] = auditResult{Pair: p, Verdict: v}
		}(i, p)
	}
	wg.Wait()
	for _, e := range errs {
		if e != nil {
			return nil, e
		}
	}
	falseN := 0
	for _, r := range results {
		mark := "OK   "
		if !r.Verdict.Correct {
			falseN++
			mark = "FALSE"
		}
		fmt.Printf("%s old=%d new=%d lost=%v | %s\n", mark, r.Pair.OldID, r.Pair.NewID, r.Verdict.InfoLost, r.Verdict.Reason)
	}
	fmt.Printf("\naudited %d pairs | false supersessions: %d (%.1f%%)\n",
		len(results), falseN, 100*float64(falseN)/float64(len(results)))
	return results, nil
}
