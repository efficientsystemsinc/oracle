package truth

// Predicate canonicalization: the enrichment LLM mints a long tail of
// near-synonym and sentence-length predicates (30k+ names for what is really
// a compact verb set). `oracle canonpreds` asks the frontier LLM for a canonical
// mapping and (with --apply) folds the tail: triples repointed, alias rows
// recorded so storeTriples never resurrects a folded name, seen_counts
// merged, orphaned predicates deleted — one transaction.

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"sort"
	"strings"
	"sync"
)

const canonPrompt = `You canonicalize predicate names for an engineering knowledge graph.
You are given a numbered list of snake_case predicates with usage counts, plus an ANCHOR list of the most-used predicates in the graph.
For EVERY input predicate, decide its canonical form:
- Keep genuinely distinct relations as themselves (map to self).
- Fold synonyms and trivial variants into one general verb (e.g. include/includes/included_in -> includes; hosted_on/deployed_on/running_on -> runs_on).
- Fold over-specific or sentence-length predicates into the nearest general verb (e.g. has_ast_chunks_but_zero_graph_extraction_support_in -> lacks_support_for or supports; pick the closest general relation).
- Prefer a name from the ANCHOR list whenever it fits; otherwise a short snake_case verb (1-3 words).
Canonical names must be snake_case, <= 3 words. Return JSON: {"map": {"input_predicate": "canonical", ...}} with an entry for EVERY input predicate.`

type predRow struct {
	id   int64
	name string
	seen int64
}

// canonBatch asks the LLM for one batch's mapping. anchors ground every batch
// in the same target vocabulary so parallel batches converge.
func canonBatch(batch []predRow, anchors []string) (map[string]string, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "ANCHORS: %s\n\nPREDICATES:\n", strings.Join(anchors, ", "))
	for _, p := range batch {
		fmt.Fprintf(&b, "%s (seen %d)\n", p.name, p.seen)
	}
	var out struct {
		Map map[string]string `json:"map"`
	}
	if err := llm.ChatJSON(canonPrompt, b.String(), 16000, &out); err != nil {
		return nil, err
	}
	if len(out.Map) == 0 {
		return nil, fmt.Errorf("canon batch returned empty map for %d predicates", len(batch))
	}
	return out.Map, nil
}

// buildCanonMapping fans all predicates to the LLM and returns alias -> canonical
// (self-maps removed, chains resolved to their terminal canonical).
func buildCanonMapping(db *sql.DB) (map[string]string, error) {
	preds, err := loadPredicates(db)
	if err != nil {
		return nil, err
	}
	sort.Slice(preds, func(i, j int) bool { return preds[i].seen > preds[j].seen })
	var anchors []string
	for i := 0; i < min(150, len(preds)); i++ {
		anchors = append(anchors, preds[i].name)
	}
	const batchSize = 200
	var batches [][]predRow
	for i := 0; i < len(preds); i += batchSize {
		batches = append(batches, preds[i:min(i+batchSize, len(preds))])
	}
	fmt.Printf("canonpreds: %d predicates in %d batches\n", len(preds), len(batches))

	raw := map[string]string{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error
	sem := make(chan struct{}, 8)
	for bi, batch := range batches {
		wg.Add(1)
		go func(bi int, batch []predRow) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			mu.Lock()
			stop := firstErr != nil
			mu.Unlock()
			if stop {
				return
			}
			m, err := canonBatch(batch, anchors)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("batch %d: %w", bi, err)
				}
				return
			}
			for k, v := range m {
				raw[k] = v
			}
		}(bi, batch)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return resolveCanonChains(raw), nil
}

// resolveCanonChains normalizes LLM output into a flat alias -> terminal
// canonical map: canonical names are cleaned to snake_case, chains
// (a -> b -> c) are followed to their end, cycles break at the entry point,
// and self-maps drop out.
func resolveCanonChains(raw map[string]string) map[string]string {
	clean := map[string]string{}
	for k, v := range raw {
		c := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(v), " ", "_"))
		if c == "" {
			continue
		}
		clean[k] = c
	}
	out := map[string]string{}
	for k := range clean {
		c := clean[k]
		for hops := 0; hops < 10; hops++ {
			next, ok := clean[c]
			if !ok || next == c {
				break
			}
			if next == k { // cycle: keep k as its own canonical
				c = k
				break
			}
			c = next
		}
		if c != k {
			out[k] = c
		}
	}
	return out
}

func loadPredicates(db *sql.DB) ([]predRow, error) {
	rows, err := db.Query("SELECT id, name, seen_count FROM predicates")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var preds []predRow
	for rows.Next() {
		var p predRow
		if err := rows.Scan(&p.id, &p.name, &p.seen); err != nil {
			return nil, err
		}
		preds = append(preds, p)
	}
	return preds, rows.Err()
}

// applyCanonMapping folds aliases into canonicals in ONE transaction:
// repoint triples, record alias rows, merge seen_counts, delete orphans.
// Pure fold mechanics — no LLM — so it is unit-testable with a fixed mapping.
func applyCanonMapping(db *sql.DB, mapping map[string]string) (folded int, err error) {
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	for alias, canon := range mapping {
		if alias == canon {
			return folded, fmt.Errorf("self-map %q leaked into apply", alias)
		}
		var aliasID int64
		err := tx.QueryRow("SELECT id FROM predicates WHERE name = ?", alias).Scan(&aliasID)
		if err == sql.ErrNoRows {
			continue // predicate vanished (or was folded via a prior run); nothing to move
		}
		if err != nil {
			return folded, err
		}
		var canonID int64
		if err := tx.QueryRow("SELECT id FROM predicates WHERE name = ?", canon).Scan(&canonID); err == sql.ErrNoRows {
			res, err := tx.Exec("INSERT INTO predicates(name, seen_count) VALUES(?,0)", canon)
			if err != nil {
				return folded, err
			}
			canonID, _ = res.LastInsertId()
		} else if err != nil {
			return folded, err
		}
		if _, err := tx.Exec("UPDATE triples SET predicate_id = ? WHERE predicate_id = ?", canonID, aliasID); err != nil {
			return folded, err
		}
		if _, err := tx.Exec(`INSERT INTO predicate_aliases(alias, predicate_id) VALUES(?,?)
			ON CONFLICT(alias) DO UPDATE SET predicate_id = excluded.predicate_id`, alias, canonID); err != nil {
			return folded, err
		}
		if _, err := tx.Exec(`UPDATE predicates SET seen_count = seen_count +
			(SELECT seen_count FROM predicates WHERE id = ?) WHERE id = ?`, aliasID, canonID); err != nil {
			return folded, err
		}
		if _, err := tx.Exec("DELETE FROM predicates WHERE id = ?", aliasID); err != nil {
			return folded, err
		}
		folded++
	}
	return folded, tx.Commit()
}

// canonPreds is the `oracle canonpreds [--apply]` entrypoint.
func CanonPreds(db *sql.DB, apply bool) error {
	var before int
	if err := db.QueryRow("SELECT COUNT(*) FROM predicates").Scan(&before); err != nil {
		return err
	}
	mapping, err := buildCanonMapping(db)
	if err != nil {
		return err
	}
	// print grouped: canonical <- aliases (largest groups first)
	groups := map[string][]string{}
	for a, c := range mapping {
		groups[c] = append(groups[c], a)
	}
	var canons []string
	for c := range groups {
		canons = append(canons, c)
	}
	sort.Slice(canons, func(i, j int) bool { return len(groups[canons[i]]) > len(groups[canons[j]]) })
	for _, c := range canons {
		sort.Strings(groups[c])
		fmt.Printf("%s <- %s\n", c, strings.Join(groups[c], ", "))
	}
	fmt.Printf("\n%d predicates, %d to fold into %d canonicals\n", before, len(mapping), len(groups))
	if !apply {
		fmt.Println("dry run — pass --apply to fold")
		return nil
	}
	folded, err := applyCanonMapping(db, mapping)
	if err != nil {
		return err
	}
	var after int
	if err := db.QueryRow("SELECT COUNT(*) FROM predicates").Scan(&after); err != nil {
		return err
	}
	fmt.Printf("applied: folded %d, predicates %d -> %d\n", folded, before, after)
	return nil
}
