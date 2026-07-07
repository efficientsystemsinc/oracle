package serve

// HTTP surface + ingest loop. `oracle up` runs both.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"oracle/internal/ask"
	"oracle/internal/ingest"
	"oracle/internal/kb"
	"oracle/internal/search"
	"oracle/internal/store"
	"strconv"
	"sync/atomic"
	"time"
)

const ingestInterval = 300 * time.Second

var loopRunning atomic.Bool

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, err error) {
	http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), 500)
}

func qInt(r *http.Request, name string, def int) int {
	if v := r.URL.Query().Get(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func qFloat(r *http.Request, name string, def float64) float64 {
	if v := r.URL.Query().Get(name); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func ServeHTTP(db *sql.DB, port int, maxCalls int, runLoop bool) error {
	// GPU vector store: load all corpus vectors into the MLX engine once so
	// the per-query cosine arms run on Metal (ORACLE_MLX=1; ORACLE_MLX_VECS=0
	// opts out). Failure is fatal — no silent fallback to the slow path.
	if err := search.InitVecStore(db); err != nil {
		return fmt.Errorf("serve: init GPU vector store: %w", err)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		var live, total, files int
		_ = db.QueryRow("SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL").Scan(&live)
		_ = db.QueryRow("SELECT COUNT(*) FROM facts").Scan(&total)
		_ = db.QueryRow("SELECT COUNT(*) FROM files").Scan(&files)
		var lastCycle string
		_ = db.QueryRow("SELECT v FROM meta WHERE k='last_cycle'").Scan(&lastCycle)
		var lc any
		_ = json.Unmarshal([]byte(lastCycle), &lc)
		writeJSON(w, map[string]any{"ok": true, "live_facts": live, "total_facts": total,
			"files_tracked": files, "last_cycle": lc, "ingest_loop": loopRunning.Load(),
			"latency_ms": store.LatencySnapshot()})
	})

	mux.HandleFunc("GET /query", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		defer store.RecordLatency("query", t0)
		q := r.URL.Query().Get("q")
		if q == "" {
			http.Error(w, `{"error":"q required"}`, 400)
			return
		}
		hits, err := search.Search(db, q, r.URL.Query().Get("repo"), qInt(r, "k", 10), qFloat(r, "as_of", 0), true)
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, map[string]any{"q": q, "hits": hits})
	})

	mux.HandleFunc("GET /ask", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		if q == "" {
			http.Error(w, `{"error":"q required"}`, 400)
			return
		}
		answer, hits, conf, err := ask.AskAuto(db, q, r.URL.Query().Get("repo"), qFloat(r, "as_of", 0))
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, map[string]any{"q": q, "answer": answer, "sources": hits,
			"confidence": conf.Score, "abstained": conf.Abstained, "features": conf.Features})
	})

	mux.HandleFunc("GET /brief", func(w http.ResponseWriter, r *http.Request) {
		b, err := search.Brief(db, r.URL.Query().Get("repo"), qInt(r, "k", 30))
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, map[string]any{"repo": r.URL.Query().Get("repo"), "brief": b})
	})

	mux.HandleFunc("GET /entity/{name}", func(w http.ResponseWriter, r *http.Request) {
		v, err := search.EntityView(db, r.PathValue("name"), qInt(r, "k", 20))
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, v)
	})

	mux.HandleFunc("GET /facts/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
		row := db.QueryRow(`SELECT id, statement, kind, COALESCE(repo,''), entities, files, confidence,
			mass, recorded_at, valid_from, COALESCE(superseded_at,0), COALESCE(superseded_by,0),
			COALESCE(src_path,''), COALESCE(src_session,''), use_count FROM facts WHERE id = ?`, id)
		var f search.FactRow
		var supAt float64
		var supBy int64
		var srcPath, srcSession string
		var useCount int
		if err := row.Scan(&f.ID, &f.Statement, &f.Kind, &f.Repo, &f.Entities, &f.Files, &f.Confidence,
			&f.Mass, &f.RecordedAt, &f.ValidFrom, &supAt, &supBy, &srcPath, &srcSession, &useCount); err != nil {
			http.Error(w, `{"error":"not found"}`, 404)
			return
		}
		out := search.ToFactOut(f, 0, float64(time.Now().Unix()), srcSession)
		writeJSON(w, map[string]any{"fact": out, "src_path": srcPath, "use_count": useCount,
			"superseded_at": supAt, "superseded_by": supBy})
	})

	mux.HandleFunc("GET /graph/{name}", func(w http.ResponseWriter, r *http.Request) {
		v, err := kb.Traverse(db, r.PathValue("name"), qInt(r, "hops", 2), qInt(r, "limit", 60))
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, v)
	})

	mux.HandleFunc("GET /metrics/{name}", func(w http.ResponseWriter, r *http.Request) {
		v, err := ingest.MetricSeries(db, r.PathValue("name"), r.URL.Query().Get("entity"))
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, map[string]any{"metric": r.PathValue("name"), "series": v})
	})

	mux.HandleFunc("GET /narrative/{name}", func(w http.ResponseWriter, r *http.Request) {
		n, err := kb.Narrative(db, r.PathValue("name"))
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, map[string]any{"subject": r.PathValue("name"), "narrative": n})
	})

	mux.HandleFunc("POST /cycle", func(w http.ResponseWriter, r *http.Request) {
		st, err := search.Cycle(db, qInt(r, "max_calls", maxCalls), qFloat(r, "since_days", 7), qInt(r, "workers", 4), r.URL.Query().Get("force_retry") == "1")
		if err != nil {
			writeErr(w, err)
			return
		}
		writeJSON(w, st)
	})

	if runLoop {
		go func() {
			loopRunning.Store(true)
			for {
				if _, err := search.Cycle(db, maxCalls, 7, 4, false); err != nil {
					log.Printf("cycle error: %v", err)
				}
				time.Sleep(ingestInterval)
			}
		}()
	}

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	log.Printf("oracle listening on http://%s", addr)
	return http.ListenAndServe(addr, mux)
}
