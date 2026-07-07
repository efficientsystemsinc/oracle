package truth

// oracle sweep: missed-supersession finder. Recurring graph hygiene pass —
// candidate pairs come from pure SQL + vectors (no LLM), the upgraded judge
// (dates + evidence tiers) decides supersedes/contradicts/nothing, and
// verdicts are applied transactionally alongside sweep_done bookkeeping.

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	sweepMinDayGap  = 3.0  // valid_from gap below which a pair is not interesting
	sweepMinCosine  = 0.60 // embedding similarity floor for candidacy
	sweepBatchPairs = 6    // pairs per judge LLM call
)

const sweepSchema = `
CREATE TABLE IF NOT EXISTS sweep_done(
  a INTEGER NOT NULL,
  b INTEGER NOT NULL,
  verdict TEXT NOT NULL,
  judged_at REAL NOT NULL,
  PRIMARY KEY(a, b)
);`

const sweepJudgePrompt = `You maintain a bi-temporal fact graph. Each PAIR below is two facts about the same entity: OLD became true earlier (valid_from date shown) and NEW became true later. Evidence tiers: verified (transcript-confirmed) > asserted (stated once) > reported (secondhand).

For each pair pick exactly one verdict:
- "supersedes": NEW replaces/outdates OLD — the same claim updated to a newer state, or a duplicate restated later. The newer fact supersedes the older one, never the reverse.
- "contradicts": OLD and NEW conflict while both claim to be current — not a clean replacement.
- "nothing": different aspects of the same system, unrelated claims, or both can be true simultaneously.

Weigh dates and evidence: a verified OLD fact should only be superseded when NEW clearly covers the same ground; mere topical similarity is "nothing".
Return JSON: {"verdicts":[{"pair":int, "verdict":"supersedes|contradicts|nothing"}]} with one entry for every pair.`

// sweepPair: a candidate pair; Old always has the strictly earlier valid_from.
type sweepPair struct {
	Old, New         int64
	OldVF, NewVF     float64
	OldStmt, NewStmt string
	OldEv, NewEv     string
	Kind             string // pairs are same-kind by construction
	Sim              float64
}

func (p sweepPair) key() [2]int64 { // canonical a<b ordering for sweep_done
	if p.Old < p.New {
		return [2]int64{p.Old, p.New}
	}
	return [2]int64{p.New, p.Old}
}

// sweepCandidates: for each entity, pair its LIVE facts within the same kind
// whose valid_from differ by > 3 days and whose vectors cosine > 0.60.
// Pairs already edged (any type, either direction) or already in sweep_done
// are skipped. Per-entity cap prefers highest similarity; global cap likewise.
func sweepCandidates(db *sql.DB, perEntity, maxPairs int) ([]sweepPair, error) {
	vecs := map[int64][]float32{}
	vrows, err := db.Query(`SELECT v.fact_id, v.vec FROM fact_vecs v
		JOIN facts f ON f.id = v.fact_id WHERE f.superseded_at IS NULL`)
	if err != nil {
		return nil, err
	}
	for vrows.Next() {
		var id int64
		var blob []byte
		if err := vrows.Scan(&id, &blob); err != nil {
			vrows.Close()
			return nil, err
		}
		vecs[id] = store.BlobToVec(blob)
	}
	vrows.Close()
	if err := vrows.Err(); err != nil {
		return nil, err
	}

	edged := map[[2]int64]bool{}
	erows, err := db.Query(`SELECT src, dst FROM edges
		UNION SELECT a, b FROM sweep_done`)
	if err != nil {
		return nil, err
	}
	for erows.Next() {
		var s, d int64
		if err := erows.Scan(&s, &d); err != nil {
			erows.Close()
			return nil, err
		}
		if s > d {
			s, d = d, s
		}
		edged[[2]int64{s, d}] = true
	}
	erows.Close()
	if err := erows.Err(); err != nil {
		return nil, err
	}

	type factRow struct {
		id        int64
		kind      string
		validFrom float64
		stmt, ev  string
	}
	byEntity := map[int64][]factRow{}
	var entityOrder []int64
	frows, err := db.Query(`SELECT fe.entity_id, f.id, f.kind, f.valid_from, f.statement, f.evidence
		FROM fact_entities fe JOIN facts f ON f.id = fe.fact_id
		WHERE f.superseded_at IS NULL ORDER BY fe.entity_id, f.valid_from`)
	if err != nil {
		return nil, err
	}
	for frows.Next() {
		var eid int64
		var fr factRow
		if err := frows.Scan(&eid, &fr.id, &fr.kind, &fr.validFrom, &fr.stmt, &fr.ev); err != nil {
			frows.Close()
			return nil, err
		}
		if _, seen := byEntity[eid]; !seen {
			entityOrder = append(entityOrder, eid)
		}
		byEntity[eid] = append(byEntity[eid], fr)
	}
	frows.Close()
	if err := frows.Err(); err != nil {
		return nil, err
	}

	seen := map[[2]int64]bool{} // a pair can surface via several shared entities
	var all []sweepPair
	for _, eid := range entityOrder {
		facts := byEntity[eid]
		var entPairs []sweepPair
		for i := 0; i < len(facts); i++ {
			for j := i + 1; j < len(facts); j++ {
				a, b := facts[i], facts[j]
				if a.kind != b.kind {
					continue
				}
				gap := b.validFrom - a.validFrom // rows are valid_from-ordered
				if gap <= sweepMinDayGap*86400 {
					continue
				}
				va, vb := vecs[a.id], vecs[b.id]
				if va == nil || vb == nil {
					continue
				}
				sim := store.Dot(va, vb) // vectors are unit-normalized
				if sim <= sweepMinCosine {
					continue
				}
				p := sweepPair{Old: a.id, New: b.id, OldVF: a.validFrom, NewVF: b.validFrom,
					OldStmt: a.stmt, NewStmt: b.stmt, OldEv: a.ev, NewEv: b.ev, Kind: a.kind, Sim: sim}
				if edged[p.key()] {
					continue
				}
				entPairs = append(entPairs, p)
			}
		}
		sort.Slice(entPairs, func(i, j int) bool { return entPairs[i].Sim > entPairs[j].Sim })
		if len(entPairs) > perEntity {
			entPairs = entPairs[:perEntity]
		}
		for _, p := range entPairs {
			if !seen[p.key()] {
				seen[p.key()] = true
				all = append(all, p)
			}
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Sim > all[j].Sim })
	if len(all) > maxPairs {
		all = all[:maxPairs]
	}
	return all, nil
}

// sweepJudgeBatch sends one batch of pairs to the judge. Returns verdict per
// batch-local index; every pair gets a verdict ("nothing" if the model omits it
// is NOT assumed — missing entries are an error, failures stay loud).
// ORACLE_LOCAL_JUDGE shadow/active applies here exactly as in ingest
// (localjudge.go); db is used only for judge_shadow logging.
func sweepJudgeBatch(db DBE, batch []sweepPair) ([]string, error) {
	mode, err := LocalJudgeMode()
	if err != nil {
		return nil, err
	}
	if mode != LocalJudgeOff {
		if err := EnsureJudgeShadow(db); err != nil {
			return nil, err
		}
	}
	sides := func(p sweepPair) (string, string) {
		return RenderJudgeSide(p.OldStmt, store.AsOfDate(p.OldVF), p.OldEv, p.Kind),
			RenderJudgeSide(p.NewStmt, store.AsOfDate(p.NewVF), p.NewEv, p.Kind)
	}

	verdicts := make([]string, len(batch))
	llmIdx := make([]int, 0, len(batch)) // batch indexes still needing the frontier LLM
	if mode == LocalJudgeActive {
		threshold, err := LocalJudgeMargin()
		if err != nil {
			return nil, err
		}
		now := float64(time.Now().Unix())
		for i, p := range batch {
			oldSide, newSide := sides(p)
			v, ok, err := ActiveJudgePair(db, now, p.Old, p.New, threshold, oldSide, newSide)
			if err != nil {
				return nil, err
			}
			if ok {
				verdicts[i] = v
			} else {
				llmIdx = append(llmIdx, i)
			}
		}
		if len(llmIdx) == 0 {
			return verdicts, nil
		}
		sub := make([]sweepPair, len(llmIdx))
		for j, i := range llmIdx {
			sub[j] = batch[i]
		}
		subV, err := sweepLLMBatch(sub)
		if err != nil {
			return nil, err
		}
		for j, i := range llmIdx {
			verdicts[i] = subV[j]
		}
		return verdicts, nil
	}

	verdicts, err = sweepLLMBatch(batch)
	if err != nil {
		return nil, err
	}
	if mode == LocalJudgeShadow {
		now := float64(time.Now().Unix())
		for i, p := range batch {
			oldSide, newSide := sides(p)
			ShadowJudgePair(db, now, p.Old, p.New, verdicts[i], oldSide, newSide)
		}
	}
	return verdicts, nil
}

// sweepLLMBatch is the original the frontier LLM batch call, untouched.
func sweepLLMBatch(batch []sweepPair) ([]string, error) {
	var b strings.Builder
	for i, p := range batch {
		fmt.Fprintf(&b, "PAIR %d:\n  OLD (id %d, %s, %s): %s\n  NEW (id %d, %s, %s): %s\n",
			i, p.Old, store.AsOfDate(p.OldVF), p.OldEv, p.OldStmt,
			p.New, store.AsOfDate(p.NewVF), p.NewEv, p.NewStmt)
	}
	var out struct {
		Verdicts []struct {
			Pair    int    `json:"pair"`
			Verdict string `json:"verdict"`
		} `json:"verdicts"`
	}
	if err := llm.ChatJSON(sweepJudgePrompt, b.String(), 4000, &out); err != nil {
		return nil, err
	}
	verdicts := make([]string, len(batch))
	for _, v := range out.Verdicts {
		if v.Pair < 0 || v.Pair >= len(batch) {
			continue
		}
		switch v.Verdict {
		case "supersedes", "contradicts", "nothing":
			verdicts[v.Pair] = v.Verdict
		}
	}
	for i, v := range verdicts {
		if v == "" {
			return nil, fmt.Errorf("judge returned no verdict for pair %d (old=%d new=%d)", i, batch[i].Old, batch[i].New)
		}
	}
	return verdicts, nil
}

// sweepApply writes one verdict transactionally with its sweep_done row.
// supersedes: stamp superseded_at/superseded_by NOW + edge. contradicts: edge
// only. nothing: sweep_done only. Newer always supersedes older, never reverse.
func sweepApply(db *sql.DB, p sweepPair, verdict string) error {
	now := float64(time.Now().Unix())
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	switch verdict {
	case "supersedes":
		if _, err := tx.Exec(`UPDATE facts SET superseded_at = ?, superseded_by = ?
			WHERE id = ? AND superseded_at IS NULL`, now, p.New, p.Old); err != nil {
			return err
		}
		if _, err := tx.Exec(`INSERT OR IGNORE INTO edges(src, dst, type, recorded_at)
			VALUES(?,?,?,?)`, p.New, p.Old, "supersedes", now); err != nil {
			return err
		}
	case "contradicts":
		if _, err := tx.Exec(`INSERT OR IGNORE INTO edges(src, dst, type, recorded_at)
			VALUES(?,?,?,?)`, p.New, p.Old, "contradicts", now); err != nil {
			return err
		}
	case "nothing":
		// bookkeeping only
	default:
		return fmt.Errorf("unknown verdict %q", verdict)
	}
	k := p.key()
	if _, err := tx.Exec(`INSERT OR IGNORE INTO sweep_done(a, b, verdict, judged_at)
		VALUES(?,?,?,?)`, k[0], k[1], verdict, now); err != nil {
		return err
	}
	return tx.Commit()
}

type sweepStats struct {
	Pairs, Superseded, Contradicted, Nothing, Errors int
}

// sweep runs the full pass: candidates -> parallel judge -> apply.
// dryRun prints verdicts and writes NOTHING (no sweep_done rows either).
func Sweep(db *sql.DB, workers, perEntity, maxPairs int, dryRun bool) (sweepStats, error) {
	if _, err := db.Exec(sweepSchema); err != nil {
		return sweepStats{}, fmt.Errorf("sweep schema: %w", err)
	}
	if workers < 1 {
		workers = 1
	}
	pairs, err := sweepCandidates(db, perEntity, maxPairs)
	if err != nil {
		return sweepStats{}, err
	}
	fmt.Printf("sweep: %d candidate pairs\n", len(pairs))
	st := sweepStats{Pairs: len(pairs)}
	if len(pairs) == 0 {
		return st, nil
	}

	batches := make(chan []sweepPair)
	var mu sync.Mutex // guards st + progress counter; db writes serialize on the single sqlite conn
	var firstErr error
	done := 0
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range batches {
				verdicts, err := sweepJudgeBatch(db, batch)
				mu.Lock()
				if err != nil {
					st.Errors += len(batch)
					done += len(batch)
					if firstErr == nil {
						firstErr = err
					}
					fmt.Printf("sweep: judge batch failed: %v\n", err)
					mu.Unlock()
					continue
				}
				mu.Unlock()
				for i, p := range batch {
					v := verdicts[i]
					if dryRun {
						fmt.Printf("[dry-run] %-11s old=%d new=%d sim=%.3f\n  OLD: %s\n  NEW: %s\n",
							v, p.Old, p.New, p.Sim, store.Truncate(p.OldStmt, 140), store.Truncate(p.NewStmt, 140))
					} else if err := sweepApply(db, p, v); err != nil {
						mu.Lock()
						st.Errors++
						if firstErr == nil {
							firstErr = err
						}
						mu.Unlock()
						fmt.Printf("sweep: apply old=%d new=%d failed: %v\n", p.Old, p.New, err)
						continue
					}
					mu.Lock()
					switch v {
					case "supersedes":
						st.Superseded++
					case "contradicts":
						st.Contradicted++
					default:
						st.Nothing++
					}
					done++
					if done%100 == 0 {
						fmt.Printf("sweep: %d/%d judged (sup=%d con=%d none=%d err=%d)\n",
							done, len(pairs), st.Superseded, st.Contradicted, st.Nothing, st.Errors)
					}
					mu.Unlock()
				}
			}
		}()
	}
	for i := 0; i < len(pairs); i += sweepBatchPairs {
		end := min(i+sweepBatchPairs, len(pairs))
		batches <- pairs[i:end]
	}
	close(batches)
	wg.Wait()
	if firstErr != nil && st.Superseded+st.Contradicted+st.Nothing == 0 {
		return st, fmt.Errorf("all judge batches failed: %w", firstErr)
	}
	fmt.Printf("sweep done: %d pairs -> %d superseded, %d contradicted, %d nothing, %d errors%s\n",
		st.Pairs, st.Superseded, st.Contradicted, st.Nothing, st.Errors,
		map[bool]string{true: " (dry-run, nothing written)", false: ""}[dryRun])
	return st, nil
}
