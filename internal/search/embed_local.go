package search

// Local-embedder swap (ORACLE_LOCAL_EMBED=1): routes embedTexts to the local
// ONNX model (embedLocal, see infer_stub.go) and points the cosine arms at
// the fact_vecs_local / fact_para_vecs_local tables, which are populated by
// `oracle reembed`. Query vectors and corpus vectors MUST come from the same
// model — mixing models breaks cosine — so reads fail loudly if the flag is
// on but the _local tables are empty.

import (
	"database/sql"
	"fmt"
	"oracle/internal/store"
	"time"
)

// factVecsTable / paraVecsTable pick the corpus tables matching the active
// embedding backend, so query and corpus vectors always share one model.
func factVecsTable() string {
	if store.LocalEmbedEnabled() {
		return "fact_vecs_local"
	}
	return "fact_vecs"
}

func paraVecsTable() string {
	if store.LocalEmbedEnabled() {
		return "fact_para_vecs_local"
	}
	return "fact_para_vecs"
}

// checkLocalVecs guards the read path: ORACLE_LOCAL_EMBED=1 with an empty
// fact_vecs_local means the corpus hasn't been migrated — searching would
// silently return nothing (or, worse, mixed-model cosine). Fail loudly.
func checkLocalVecs(db *sql.DB) error {
	if !store.LocalEmbedEnabled() {
		return nil
	}
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM fact_vecs_local").Scan(&n); err != nil {
		return fmt.Errorf("ORACLE_LOCAL_EMBED=1 but fact_vecs_local is unreadable: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("ORACLE_LOCAL_EMBED=1 but fact_vecs_local is empty — run `oracle reembed` first (query and corpus vectors must come from the same model)")
	}
	return nil
}

// reembed re-embeds ALL live facts and their paraphrases through embedTexts
// (so it uses whichever backend the flag selects — normally run with
// ORACLE_LOCAL_EMBED=1) into fact_vecs_local / fact_para_vecs_local.
// Resumable: rows already present in the _local tables are skipped.
func Reembed(db *sql.DB, batch int) error {
	if batch <= 0 {
		return fmt.Errorf("reembed: batch must be > 0")
	}
	start := time.Now()
	nf, err := reembedLoop(db, batch, "facts",
		`SELECT f.id, f.statement FROM facts f
		 LEFT JOIN fact_vecs_local v ON v.fact_id = f.id
		 WHERE f.superseded_at IS NULL AND v.fact_id IS NULL LIMIT ?`,
		"INSERT OR REPLACE INTO fact_vecs_local(fact_id, vec) VALUES(?,?)")
	if err != nil {
		return err
	}
	np, err := reembedLoop(db, batch, "paraphrases",
		`SELECT p.fact_id, p.text FROM fact_paraphrases p
		 JOIN facts f ON f.id = p.fact_id
		 LEFT JOIN fact_para_vecs_local v ON v.fact_id = p.fact_id
		 WHERE f.superseded_at IS NULL AND v.fact_id IS NULL
		 GROUP BY p.fact_id LIMIT ?`,
		"INSERT OR REPLACE INTO fact_para_vecs_local(fact_id, vec) VALUES(?,?)")
	if err != nil {
		return err
	}
	fmt.Printf("reembed done: %d facts + %d paraphrases in %.0fs\n",
		nf, np, time.Since(start).Seconds())
	return nil
}

// reembedLoop embeds (id, text) rows from selectSQL in batches, writing each
// batch transactionally via insertSQL, until the select returns no rows.
// Because selectSQL excludes already-written ids, an interrupted run resumes
// where it stopped.
func reembedLoop(db *sql.DB, batch int, label, selectSQL, insertSQL string) (int, error) {
	total := 0
	for {
		rows, err := db.Query(selectSQL, batch)
		if err != nil {
			return total, err
		}
		var ids []int64
		var texts []string
		for rows.Next() {
			var id int64
			var s string
			if err := rows.Scan(&id, &s); err != nil {
				rows.Close()
				return total, err
			}
			ids = append(ids, id)
			texts = append(texts, s)
		}
		rows.Close()
		if len(ids) == 0 {
			return total, nil
		}
		vecs, err := embedTexts(texts)
		if err != nil {
			return total, fmt.Errorf("reembed %s (after %d): %w", label, total, err)
		}
		tx, err := db.Begin()
		if err != nil {
			return total, err
		}
		for i, id := range ids {
			if _, err := tx.Exec(insertSQL, id, store.VecToBlob(vecs[i])); err != nil {
				tx.Rollback()
				return total, err
			}
		}
		if err := tx.Commit(); err != nil {
			return total, err
		}
		total += len(ids)
		fmt.Printf("reembed %s: %d done\n", label, total)
		if len(ids) < batch {
			return total, nil
		}
	}
}
