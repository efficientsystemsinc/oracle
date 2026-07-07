package search

// Paraphrase index: one LLM-generated alternate phrasing per live fact, stored
// as (a) text in fact_paraphrases + a contentful FTS5 table para_fts (third
// RRF arm in search), and (b) a vector in fact_para_vecs (merged into the
// cosine arm as an alternate surface form of the same fact). Coverage is
// mass-ordered: the most valuable facts get paraphrases first.

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"sort"
	"strings"
	"time"
)

// decayedMassOrder ranks live facts by present-day decayed mass, in SQL, so
// the paraphrase budget covers the most-massive facts first. Mirrors massNow.
const decayedMassOrder = `f.mass * exp(-0.693 * max(0.0, ? - f.valid_from) /
	(CASE f.kind WHEN 'status' THEN 7.0 WHEN 'todo' THEN 14.0 WHEN 'decision' THEN 120.0
	 WHEN 'gotcha' THEN 120.0 WHEN 'preference' THEN 365.0 ELSE 60.0 END * 86400.0)) DESC`

const paraphraseSystem = `You rewrite engineering fact statements. The user message is JSON: {"statements": [str, ...]}. For EACH element of "statements" produce EXACTLY ONE natural alternate phrasing: different vocabulary and sentence structure, identical meaning. A statement is one element no matter how many clauses or numbers it contains — never split or merge. Keep every entity name, hostname, path, number, version, and identifier VERBATIM. Do not add or drop information. Return JSON: {"p": [str, ...]} — same order and same count as "statements".`

const paraBatchSize = 20

// errParaShape marks LLM output-shape failures (wrong count, empty entry) —
// retryable per statement, unlike transport/auth errors which must stay loud.
var errParaShape = errors.New("paraphrase shape")

// paraphraseBatch asks the LLM for one paraphrase per statement. Loud on any
// shape mismatch — a silent partial write would poison coverage accounting.
func paraphraseBatch(stmts []string) ([]string, error) {
	in, err := json.Marshal(map[string]any{"statements": stmts})
	if err != nil {
		return nil, err
	}
	var out struct {
		P []string `json:"p"`
	}
	if err := llm.ChatJSON(paraphraseSystem, string(in), 8000, &out); err != nil {
		return nil, err
	}
	if len(out.P) != len(stmts) {
		return nil, fmt.Errorf("%w: count mismatch: got %d want %d", errParaShape, len(out.P), len(stmts))
	}
	for i, p := range out.P {
		if strings.TrimSpace(p) == "" {
			return nil, fmt.Errorf("%w: empty paraphrase at position %d", errParaShape, i)
		}
	}
	return out.P, nil
}

// paraphraseRun generates paraphrases for up to maxCalls batches of live
// facts without one, most-massive first. Each batch: LLM call + embed call +
// one tx (texts, FTS rows, vecs, done marks commit atomically).
func ParaphraseRun(db *sql.DB, maxCalls int) (facts int, calls int, err error) {
	return paraphraseRunWith(db, maxCalls, paraphraseBatch)
}

// paraphraseRunWith is paraphraseRun with the LLM call injectable for tests.
// Shape errors (errParaShape) fall back to per-statement calls so one compound
// statement the model insists on splitting cannot wedge the whole backfill —
// mass-ordered selection would re-pick the same batch forever. A statement
// that shape-fails alone is recorded in paraphrase_skip (excluded from future
// selection, surfaced in the CLI summary). Transport/auth errors still abort.
func paraphraseRunWith(db *sql.DB, maxCalls int, batch func([]string) ([]string, error)) (facts int, calls int, err error) {
	now := float64(time.Now().Unix())
	for calls < maxCalls {
		rows, err := db.Query(`SELECT f.id, f.statement FROM facts f
			LEFT JOIN paraphrase_done d ON d.fact_id = f.id
			LEFT JOIN paraphrase_skip sk ON sk.fact_id = f.id
			WHERE f.superseded_at IS NULL AND d.fact_id IS NULL AND sk.fact_id IS NULL
			ORDER BY `+decayedMassOrder+` LIMIT ?`, now, paraBatchSize)
		if err != nil {
			return facts, calls, err
		}
		var ids []int64
		var stmts []string
		for rows.Next() {
			var id int64
			var s string
			if err := rows.Scan(&id, &s); err != nil {
				rows.Close()
				return facts, calls, err
			}
			ids = append(ids, id)
			stmts = append(stmts, s)
		}
		rows.Close()
		if len(ids) == 0 {
			return facts, calls, nil
		}
		paras, err := batch(stmts)
		calls++
		if errors.Is(err, errParaShape) {
			for i := 0; i < len(ids) && calls < maxCalls; i++ {
				p1, err1 := batch([]string{stmts[i]})
				calls++
				if errors.Is(err1, errParaShape) {
					if _, serr := db.Exec("INSERT OR IGNORE INTO paraphrase_skip(fact_id, err) VALUES(?,?)", ids[i], err1.Error()); serr != nil {
						return facts, calls, serr
					}
					continue
				}
				if err1 != nil {
					return facts, calls, fmt.Errorf("singleton after %d calls: %w", calls, err1)
				}
				if err := storeParaphrases(db, ids[i:i+1], p1); err != nil {
					return facts, calls, err
				}
				facts++
			}
			continue
		}
		if err != nil {
			return facts, calls, fmt.Errorf("batch after %d calls: %w", calls, err)
		}
		if err := storeParaphrases(db, ids, paras); err != nil {
			return facts, calls, err
		}
		facts += len(ids)
	}
	return facts, calls, nil
}

// storeParaphrases embeds and commits paraphrases for ids atomically:
// texts, FTS rows, vecs, and done marks land in one tx.
func storeParaphrases(db *sql.DB, ids []int64, paras []string) error {
	vecs, err := embedTexts(paras)
	if err != nil {
		return fmt.Errorf("embed paraphrases: %w", err)
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	for i, id := range ids {
		if _, err := tx.Exec("INSERT INTO fact_paraphrases(fact_id, text) VALUES(?,?)", id, paras[i]); err != nil {
			tx.Rollback()
			return err
		}
		if _, err := tx.Exec("INSERT INTO para_fts(text, fact_id) VALUES(?,?)", paras[i], id); err != nil {
			tx.Rollback()
			return err
		}
		if _, err := tx.Exec("INSERT OR REPLACE INTO "+paraVecsTable()+"(fact_id, vec) VALUES(?,?)", id, store.VecToBlob(vecs[i])); err != nil {
			tx.Rollback()
			return err
		}
		if _, err := tx.Exec("INSERT OR IGNORE INTO paraphrase_done(fact_id) VALUES(?)", id); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// paraFtsRanks is the lexical paraphrase arm: rank live facts whose
// paraphrase matches the query, best-bm25 first, deduped per fact.
func paraFtsRanks(db *sql.DB, q string, asOf float64) (map[int64]int, []int64, error) {
	where := "WHERE para_fts MATCH ?"
	args := []any{ftsQuery(q)}
	if asOf > 0 {
		where += " AND " + store.AsOfPredicate
		args = append(args, asOf, asOf)
	} else {
		where += " AND f.superseded_at IS NULL"
	}
	rows, err := db.Query(`SELECT f.id FROM para_fts JOIN facts f ON f.id = para_fts.fact_id `+
		where+` ORDER BY bm25(para_fts) LIMIT 60`, args...)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	rank := map[int64]int{}
	var order []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, nil, err
		}
		if _, seen := rank[id]; seen {
			continue // a fact has one paraphrase today, but stay dupe-safe
		}
		rank[id] = len(order)
		order = append(order, id)
	}
	return rank, order, nil
}

// paraCosTop returns fact_id -> similarity of the query vec against the top n
// live-fact paraphrase vectors (alternate surface form of the same fact).
func paraCosTop(db *sql.DB, qv []float32, asOf float64, n int) (map[int64]float64, error) {
	where := "WHERE f.superseded_at IS NULL"
	args := []any{}
	if asOf > 0 {
		where = "WHERE " + store.AsOfPredicate
		args = append(args, asOf, asOf)
	}
	if err := checkLocalVecs(db); err != nil {
		return nil, err
	}
	rows, err := db.Query("SELECT v.fact_id, v.vec FROM "+paraVecsTable()+" v JOIN facts f ON f.id = v.fact_id "+where, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type pair struct {
		id  int64
		sim float64
	}
	var all []pair
	for rows.Next() {
		var id int64
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, err
		}
		all = append(all, pair{id, store.Dot(qv, store.BlobToVec(blob))})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].sim > all[j].sim })
	if len(all) > n {
		all = all[:n]
	}
	m := map[int64]float64{}
	for _, p := range all {
		m[p.id] = p.sim
	}
	return m, nil
}
