package serve

// CLI one-shots pay ~2s of model load; a warm daemon is usually running.
// Route through it when reachable — same results, warm-path p99.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"oracle/internal/search"
	"time"
)

func QueryViaDaemonOrLocal(db *sql.DB, q, repo string, k int, asOf float64) ([]search.FactOut, error) {
	cl := &http.Client{Timeout: 15 * time.Second}
	u := fmt.Sprintf("http://127.0.0.1:4141/query?q=%s&repo=%s&k=%d&as_of=%f",
		url.QueryEscape(q), url.QueryEscape(repo), k, asOf)
	resp, err := cl.Get(u)
	if err == nil && resp.StatusCode == 200 {
		defer resp.Body.Close()
		var out struct {
			Hits []search.FactOut `json:"hits"`
		}
		if json.NewDecoder(resp.Body).Decode(&out) == nil {
			return out.Hits, nil
		}
	}
	if resp != nil {
		resp.Body.Close()
	}
	// daemon not up: local path (loud if local models misconfigured — unchanged semantics)
	return search.Search(db, q, repo, k, asOf, true)
}
