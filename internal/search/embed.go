package search

// Embeddings: 512 dims, float32 LE blobs in fact_vecs. Remote path hits any
// OpenAI-compatible embeddings endpoint (provider.go); ORACLE_LOCAL_EMBED=1
// uses the bundled local ONNX embedder instead — no remote config needed.
// Brute-force cosine — fine for <100k facts.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"oracle/internal/infer"
	"oracle/internal/llm"
	"oracle/internal/store"
	"sort"
)

const embedDims = 512

// embedLocalFn indirects the local backend so tests can stub it; the real
// implementation is embedLocal (infer_stub.go on this branch).
var embedLocalFn = infer.EmbedLocal

// embedTexts embeds via the configured remote endpoint by default; with
// ORACLE_LOCAL_EMBED=1 it routes to the local ONNX embedder (same 512 dims,
// L2-normalized).
func embedTexts(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if store.LocalEmbedEnabled() {
		return embedLocalFn(texts)
	}
	cfg, err := llm.EmbedConfig()
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"input": texts, "dimensions": embedDims}
	if cfg.Model != "" {
		payload["model"] = cfg.Model
	}
	body, _ := json.Marshal(payload)
	raw, err := llm.PostJSON(cfg.URL, body, cfg.Key)
	if err != nil {
		return nil, err
	}
	var er struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &er); err != nil {
		return nil, err
	}
	if len(er.Data) != len(texts) {
		return nil, fmt.Errorf("embed count mismatch: %d != %d", len(er.Data), len(texts))
	}
	out := make([][]float32, len(texts))
	for _, d := range er.Data {
		out[d.Index] = store.Normalize(d.Embedding)
	}
	return out, nil
}

// embedMissing embeds facts that have no vector yet (backfill + normal ingest tail).
func embedMissing(db *sql.DB, batch int) (int, error) {
	total := 0
	for {
		rows, err := db.Query(`SELECT f.id, f.statement FROM facts f
			LEFT JOIN `+factVecsTable()+` v ON v.fact_id = f.id WHERE v.fact_id IS NULL LIMIT ?`, batch)
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
			return total, err
		}
		tx, err := db.Begin()
		if err != nil {
			return total, err
		}
		for i, id := range ids {
			if _, err := tx.Exec("INSERT OR REPLACE INTO "+factVecsTable()+"(fact_id, vec) VALUES(?,?)",
				id, store.VecToBlob(vecs[i])); err != nil {
				tx.Rollback()
				return total, err
			}
		}
		if err := tx.Commit(); err != nil {
			return total, err
		}
		total += len(ids)
		if len(ids) < batch {
			return total, nil
		}
	}
}

// deadCosHit is a superseded fact matched by the cosine arm, with its chain link.
type deadCosHit struct {
	ID   int64
	Next int64 // superseded_by (0 if the row was orphaned)
	Sim  float64
}

// cosineDeadTop returns the top n SUPERSEDED facts by cosine similarity, with
// their superseded_by links, so search can walk them to live chain heads.
// Present-time only: as-of queries must not resurrect the future.
func cosineDeadTop(db *sql.DB, qv []float32, n int) ([]deadCosHit, error) {
	if err := checkLocalVecs(db); err != nil {
		return nil, err
	}
	rows, err := db.Query(`SELECT v.fact_id, COALESCE(f.superseded_by, 0), v.vec
		FROM ` + factVecsTable() + ` v JOIN facts f ON f.id = v.fact_id WHERE f.superseded_at IS NOT NULL`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var all []deadCosHit
	for rows.Next() {
		var h deadCosHit
		var blob []byte
		if err := rows.Scan(&h.ID, &h.Next, &blob); err != nil {
			return nil, err
		}
		h.Sim = store.Dot(qv, store.BlobToVec(blob))
		all = append(all, h)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Sim > all[j].Sim })
	if len(all) > n {
		all = all[:n]
	}
	return all, nil
}

// cosineTop returns fact_id -> similarity for the top n live facts vs query vec.
func cosineTop(db *sql.DB, qv []float32, asOf float64, n int) (map[int64]float64, []int64, error) {
	where := "WHERE f.superseded_at IS NULL"
	args := []any{}
	if asOf > 0 {
		where = "WHERE " + store.AsOfPredicate
		args = append(args, asOf, asOf)
	}
	if err := checkLocalVecs(db); err != nil {
		return nil, nil, err
	}
	rows, err := db.Query("SELECT v.fact_id, v.vec FROM "+factVecsTable()+" v JOIN facts f ON f.id = v.fact_id "+where, args...)
	if err != nil {
		return nil, nil, err
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
			return nil, nil, err
		}
		all = append(all, pair{id, store.Dot(qv, store.BlobToVec(blob))})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].sim > all[j].sim })
	if len(all) > n {
		all = all[:n]
	}
	m := map[int64]float64{}
	order := make([]int64, 0, len(all))
	for _, p := range all {
		m[p.id] = p.sim
		order = append(order, p.id)
	}
	return m, order, nil
}
