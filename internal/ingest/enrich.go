package ingest

// Enrichment: retrofit facts with typed entities, S-P-O triples, and metric
// observations. One path serves both the 33k backfilled facts and every new
// fact (cycle enriches anything not yet in enrich_done).

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"oracle/internal/kb"
	"oracle/internal/llm"
	"oracle/internal/store"
	"strings"
	"sync"
	"time"
)

const enrichPrompt = `You annotate engineering facts for a knowledge graph. For EACH numbered fact, extract:
- "entities": named things with a type from: repo, service, host, model, dataset, tool, person, org, file, metric, other
- "triples": subject-predicate-object relations that the fact STATES (predicates snake_case, e.g. runs_on, deployed_to, part_of, uses, costs, superseded_by, owned_by). Object may be an entity or a literal value ("literal": true).
- "observations": numeric measurements with a canonical metric name (snake_case), value as a number, optional unit and ISO date.
Only extract what the fact explicitly states. Empty arrays are fine.
Return JSON: {"items":[{"idx":int, "entities":[{"name":str,"type":str}], "triples":[{"subject":str,"predicate":str,"object":str,"literal":bool}], "observations":[{"metric":str,"entity":str,"value":num,"unit":str,"date":str}]}]}`

type enrichItem struct {
	Idx      int `json:"idx"`
	Entities []struct {
		Name string `json:"name"`
		Type string `json:"type"`
	} `json:"entities"`
	Triples      []kb.Triple      `json:"triples"`
	Observations []kb.Observation `json:"observations"`
}

// enrichSome annotates up to maxCalls batches of 40 unenriched facts,
// fanning batches across workers (partitioned by fact id).
func EnrichSome(db *sql.DB, maxCalls int) (int, error) {
	const workers = 6
	if maxCalls <= workers {
		return enrichWorker(db, maxCalls, 1, 0)
	}
	var mu sync.Mutex
	var wg sync.WaitGroup
	total := 0
	var firstErr error
	per := (maxCalls + workers - 1) / workers
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			n, err := enrichWorker(db, per, workers, w)
			mu.Lock()
			total += n
			if err != nil && firstErr == nil {
				firstErr = err
			}
			mu.Unlock()
		}(w)
	}
	wg.Wait()
	return total, firstErr
}

func enrichWorker(db *sql.DB, maxCalls, mod, rem int) (int, error) {
	done := 0
	for call := 0; call < maxCalls; call++ {
		rows, err := db.Query(`SELECT f.id, f.statement, f.valid_from FROM facts f
			LEFT JOIN enrich_done d ON d.fact_id = f.id
			WHERE d.fact_id IS NULL AND f.id % ? = ? ORDER BY f.id LIMIT 40`, mod, rem)
		if err != nil {
			return done, err
		}
		var ids []int64
		var validFroms []float64
		var b strings.Builder
		for rows.Next() {
			var id int64
			var s string
			var vf float64
			if err := rows.Scan(&id, &s, &vf); err != nil {
				rows.Close()
				return done, err
			}
			fmt.Fprintf(&b, "%d: %s\n", len(ids), s)
			ids = append(ids, id)
			validFroms = append(validFroms, vf)
		}
		rows.Close()
		if len(ids) == 0 {
			return done, nil
		}
		var out struct {
			Items []enrichItem `json:"items"`
		}
		if err := llm.ChatJSON(enrichPrompt, b.String(), 16000, &out); err != nil {
			return done, err
		}
		now := float64(time.Now().Unix())
		byIdx := map[int]enrichItem{}
		for _, it := range out.Items {
			byIdx[it.Idx] = it
		}
		for i, fid := range ids {
			it := byIdx[i]
			for _, e := range it.Entities {
				eid, err := kb.UpsertEntityByName(db, e.Name, e.Type, now)
				if err != nil {
					continue // junk entity name: drop it, keep the rest
				}
				_, _ = db.Exec("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES(?,?)", fid, eid)
			}
			if err := kb.StoreTriples(db, fid, it.Triples, now); err != nil {
				return done, err
			}
			if err := kb.StoreObservations(db, fid, it.Observations, validFroms[i], now); err != nil {
				return done, err
			}
			if _, err := db.Exec("INSERT OR IGNORE INTO enrich_done(fact_id) VALUES(?)", fid); err != nil {
				return done, err
			}
			done++
		}
	}
	return done, nil
}

func MetricSeries(db *sql.DB, name, entity string) ([]map[string]any, error) {
	mname := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(name), " ", "_"))
	where := "WHERE m.name = ?"
	args := []any{mname}
	if entity != "" {
		eid, _, ok := kb.ResolveEntity(db, entity)
		if !ok {
			return nil, fmt.Errorf("entity %q not found", entity)
		}
		where += " AND o.entity_id = ?"
		args = append(args, eid)
	}
	rows, err := db.Query(`SELECT o.value, COALESCE(m.unit,''), o.occurred_at,
		COALESCE(e.display,''), o.fact_id
		FROM metric_observations o JOIN metrics m ON m.id = o.metric_id
		LEFT JOIN entities e ON e.id = o.entity_id `+where+` ORDER BY o.occurred_at`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]any
	for rows.Next() {
		var v float64
		var unit, disp string
		var occ float64
		var fid sql.NullInt64
		_ = rows.Scan(&v, &unit, &occ, &disp, &fid)
		out = append(out, map[string]any{
			"value": v, "unit": unit, "entity": disp, "fact_id": fid.Int64,
			"date": store.LocalDate(occ),
		})
	}
	return out, nil
}

func MetricsList(db *sql.DB) string {
	rows, err := db.Query(`SELECT m.name, COALESCE(m.unit,''), COUNT(o.id)
		FROM metrics m LEFT JOIN metric_observations o ON o.metric_id = m.id
		GROUP BY m.id ORDER BY COUNT(o.id) DESC LIMIT 40`)
	if err != nil {
		return err.Error()
	}
	defer rows.Close()
	var b strings.Builder
	for rows.Next() {
		var n, u string
		var c int
		_ = rows.Scan(&n, &u, &c)
		fmt.Fprintf(&b, "%-32s %-8s %d obs\n", n, u, c)
	}
	return b.String()
}

func JSONPrint(v any) {
	b, _ := json.MarshalIndent(v, "", " ")
	fmt.Println(string(b))
}
