package search

// mineprobes: auto-grow the eval corpus from the graph's own structure.
// Emits eval.sh-compatible TSV (question<TAB>regex<TAB>miner<TAB>as-of) on stdout:
//   (a) supersession chains — the question is the OLD superseded statement
//       (era phrasing), the regex is built from the live head's rarest tokens;
//   (b) verified / corroborated facts across ALL repos proportionally — the
//       question is a gpt paraphrase of the statement (so the probe is not a
//       verbatim FTS gift), the regex is the statement's rarest tokens;
//   (c) adversarial time probes from repaired (UPHOLD) supersession chains —
//       for each chain BOTH a current-state row (expects the live head) and an
//       as-of row (4th column carries the date; expects the OLD fact via
//       `oracle query --as-of`).
// Every regex is rejected if it matches more than maxRegexLiveHits live facts
// (a broad regex is a coin flip, not a probe).
// Reproduce with: oracle mineprobes -n 1000 > eval/probes_mined.tsv

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
)

// maxRegexLiveHits: a probe regex matching more live facts than this is too
// generic to certify a hit.
const maxRegexLiveHits = 20

type liveFact struct {
	stmt string
	repo string
}

// loadLiveFacts loads all live (unsuperseded) statements with their repo.
func loadLiveFacts(db *sql.DB) ([]liveFact, error) {
	rows, err := db.Query("SELECT statement, COALESCE(repo,'') FROM facts WHERE superseded_at IS NULL")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []liveFact
	for rows.Next() {
		var f liveFact
		if err := rows.Scan(&f.stmt, &f.repo); err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, nil
}

// docFreq counts, over live statements, how many facts contain each token.
func docFreq(live []liveFact) map[string]int {
	df := map[string]int{}
	for _, f := range live {
		seen := map[string]bool{}
		for _, t := range tokenRe.FindAllString(f.stmt, 200) {
			lc := strings.ToLower(t)
			if !seen[lc] {
				seen[lc] = true
				df[lc]++
			}
		}
	}
	return df
}

// regexTooBroad reports whether pattern matches more than maxRegexLiveHits
// live facts. An invalid pattern is treated as too broad (skip the probe).
func regexTooBroad(pattern string, live []liveFact) bool {
	re, err := regexp.Compile("(?i)" + pattern)
	if err != nil {
		return true
	}
	hits := 0
	for _, f := range live {
		if re.MatchString(f.stmt) {
			hits++
			if hits > maxRegexLiveHits {
				return true
			}
		}
	}
	return false
}

// rareRegex picks the 2-3 rarest content tokens of a statement (in statement
// order, so `a.*b` matches the printed hit line) and joins them into a grep -E
// pattern. Returns "" when the statement has no sufficiently distinctive token.
func rareRegex(statement string, df map[string]int) string {
	type cand struct {
		tok string
		pos int
		df  int
	}
	seen := map[string]bool{}
	var cands []cand
	for pos, t := range tokenRe.FindAllString(statement, 200) {
		lc := strings.ToLower(t)
		if len(lc) < 4 || ftsStopwords[lc] || seen[lc] {
			continue
		}
		seen[lc] = true
		cands = append(cands, cand{lc, pos, df[lc]})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].df < cands[j].df })
	n := 3
	if len(cands) < n {
		n = len(cands)
	}
	if n < 2 {
		return "" // one common token is not a probe, it's a coin flip
	}
	pick := cands[:n]
	if pick[0].df > 40 { // even the rarest token is generic — skip this fact
		return ""
	}
	sort.Slice(pick, func(i, j int) bool { return pick[i].pos < pick[j].pos })
	parts := make([]string, len(pick))
	for i, c := range pick {
		parts[i] = strings.ReplaceAll(c.tok, ".", `\.`)
	}
	return strings.Join(parts, ".*")
}

func tsvSanitize(s string) string {
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}

// probe is one eval row: question, expected regex, miner tag, optional as-of date.
type probe struct {
	q, re, miner, asOf string
}

// mineChainProbes emits up to n probes from supersession chains: old-era
// phrasing must retrieve content of the live chain head.
func mineChainProbes(db *sql.DB, df map[string]int, live []liveFact, n int) ([]probe, error) {
	rows, err := db.Query(`SELECT f.id, f.statement FROM facts f
		JOIN facts s ON s.id = f.superseded_by
		WHERE s.valid_from > f.valid_from + 3600
		ORDER BY f.id DESC LIMIT ?`, n*3)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type oldrow struct {
		id   int64
		stmt string
	}
	var olds []oldrow
	for rows.Next() {
		var o oldrow
		if err := rows.Scan(&o.id, &o.stmt); err != nil {
			return nil, err
		}
		olds = append(olds, o)
	}
	var out []probe
	usedHead := map[int64]bool{}
	for _, o := range olds {
		if len(out) >= n {
			break
		}
		head := chainHead(db, o.id)
		if head == o.id || usedHead[head] {
			continue
		}
		var headStmt string
		if err := db.QueryRow("SELECT statement FROM facts WHERE id = ?", head).Scan(&headStmt); err != nil {
			return nil, err
		}
		re := rareRegex(headStmt, df)
		if re == "" || regexTooBroad(re, live) {
			continue
		}
		usedHead[head] = true
		out = append(out, probe{q: tsvSanitize(o.stmt), re: re, miner: "chain"})
	}
	return out, nil
}

const paraphrasePrompt = `You rewrite engineering facts as short natural questions a developer would type
into a memory-search tool. Keep the distinctive nouns (system names, hosts, flags, numbers) so the
question is answerable, but do NOT quote the statement verbatim — rephrase around them.
Return JSON {"probes":[{"i": <index>, "q": "<question>"}]} covering every input index.`

// paraphraseBatch turns statements into questions via chatJSON, batched.
// Returns a map from input index to question; indexes the model skipped or
// returned empty are absent.
func probeParaphraseBatch(statements []string) (map[int]string, error) {
	out := map[int]string{}
	const batch = 25
	for lo := 0; lo < len(statements); lo += batch {
		hi := min(lo+batch, len(statements))
		var b strings.Builder
		for i := lo; i < hi; i++ {
			fmt.Fprintf(&b, "[%d] %s\n", i-lo, store.Truncate(statements[i], 400))
		}
		var resp struct {
			Probes []struct {
				I int    `json:"i"`
				Q string `json:"q"`
			} `json:"probes"`
		}
		if err := llm.ChatJSON(paraphrasePrompt, b.String(), 8000, &resp); err != nil {
			return nil, fmt.Errorf("paraphrase batch %d: %w", lo/batch, err)
		}
		for _, p := range resp.Probes {
			if p.I < 0 || p.I >= hi-lo || strings.TrimSpace(p.Q) == "" {
				continue
			}
			out[lo+p.I] = tsvSanitize(p.Q)
		}
	}
	return out, nil
}

// mineParaphraseProbes emits up to n probes from verified / corroborated live
// facts, sampled across ALL repos proportionally to each repo's eligible pool,
// paraphrasing statements into questions in chatJSON batches.
func mineParaphraseProbes(db *sql.DB, df map[string]int, live []liveFact, n int) ([]probe, error) {
	// verified/corroborated first (the strongest ground truth), then fill from
	// high-confidence facts — the verified pool alone is tiny in young DBs
	rows, err := db.Query(`SELECT statement, COALESCE(repo,'') FROM facts
		WHERE superseded_at IS NULL AND (evidence = 'verified' OR corroborations >= 1 OR confidence >= 0.9)
		ORDER BY (evidence = 'verified') DESC, corroborations DESC, confidence DESC, id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type item struct{ stmt, re string }
	byRepo := map[string][]item{}
	var repoOrder []string
	total := 0
	for rows.Next() {
		var s, repo string
		if err := rows.Scan(&s, &repo); err != nil {
			return nil, err
		}
		re := rareRegex(s, df)
		if re == "" || regexTooBroad(re, live) {
			continue
		}
		if _, ok := byRepo[repo]; !ok {
			repoOrder = append(repoOrder, repo)
		}
		byRepo[repo] = append(byRepo[repo], item{s, re})
		total++
	}
	if total == 0 {
		return nil, nil
	}
	// proportional allocation: each repo contributes ~ n * its share of the
	// eligible pool (at least 1 for any repo with an eligible fact), taken in
	// priority order within the repo; the concatenation is trimmed to n.
	var items []item
	for _, repo := range repoOrder {
		pool := byRepo[repo]
		quota := (n*len(pool) + total - 1) / total // ceil
		if quota > len(pool) {
			quota = len(pool)
		}
		items = append(items, pool[:quota]...)
	}
	if len(items) > n {
		items = items[:n]
	}
	stmts := make([]string, len(items))
	for i, it := range items {
		stmts[i] = it.stmt
	}
	qs, err := probeParaphraseBatch(stmts)
	if err != nil {
		return nil, err
	}
	var out []probe
	for i, it := range items {
		q, ok := qs[i]
		if !ok {
			continue
		}
		out = append(out, probe{q: q, re: it.re, miner: "paraphrase"})
	}
	return out, nil
}

// mineTimeProbes emits adversarial bi-temporal probe PAIRS from repaired
// supersession chains (repair_done verdict UPHOLD — audited, trustworthy).
// For each chain the OLD statement is paraphrased into one question, emitted
// twice: a current-state row whose regex is built from the LIVE HEAD, and an
// as-of row (4th TSV column = a date inside the old fact's validity window)
// whose regex is built from the OLD fact. Only chains where a whole-day
// as-of boundary falls strictly inside [old.valid_from, new.valid_from)
// qualify, since --as-of takes a date.
func mineTimeProbes(db *sql.DB, df map[string]int, live []liveFact, nChains int) ([]probe, error) {
	if nChains <= 0 {
		return nil, nil
	}
	rows, err := db.Query(`SELECT f.id, f.statement, f.valid_from, s.valid_from
		FROM repair_done rd
		JOIN facts f ON f.id = rd.old_id
		JOIN facts s ON s.id = f.superseded_by
		WHERE rd.verdict = 'UPHOLD' AND f.superseded_at IS NOT NULL
		ORDER BY f.id DESC`)
	if err != nil {
		return nil, fmt.Errorf("time probes need the repair_done table (run `oracle repair` first): %w", err)
	}
	defer rows.Close()
	type chain struct {
		oldID         int64
		oldStmt       string
		oldVF, newVF  float64
		asOf          string
		oldRe, headRe string
	}
	var chains []chain
	for rows.Next() {
		var c chain
		if err := rows.Scan(&c.oldID, &c.oldStmt, &c.oldVF, &c.newVF); err != nil {
			return nil, err
		}
		chains = append(chains, c)
	}
	usedHead := map[int64]bool{}
	var picked []chain
	for _, c := range chains {
		if len(picked) >= nChains {
			break
		}
		// as-of date: the first UTC midnight after the old fact became valid.
		// It must fall strictly before the superseder's valid_from, or a
		// day-granular --as-of cannot land inside the old fact's window.
		m := time.Unix(int64(c.oldVF), 0).UTC().Truncate(24 * time.Hour).Add(24 * time.Hour)
		if float64(m.Unix()) >= c.newVF {
			continue
		}
		head := chainHead(db, c.oldID)
		if head == c.oldID || usedHead[head] {
			continue
		}
		var headStmt string
		if err := db.QueryRow("SELECT statement FROM facts WHERE id = ?", head).Scan(&headStmt); err != nil {
			return nil, err
		}
		c.headRe = rareRegex(headStmt, df)
		c.oldRe = rareRegex(c.oldStmt, df)
		if c.headRe == "" || c.oldRe == "" ||
			regexTooBroad(c.headRe, live) || regexTooBroad(c.oldRe, live) {
			continue
		}
		c.asOf = m.Format("2006-01-02")
		usedHead[head] = true
		picked = append(picked, c)
	}
	stmts := make([]string, len(picked))
	for i, c := range picked {
		stmts[i] = c.oldStmt
	}
	qs, err := probeParaphraseBatch(stmts)
	if err != nil {
		return nil, err
	}
	var out []probe
	for i, c := range picked {
		q, ok := qs[i]
		if !ok {
			continue
		}
		out = append(out,
			probe{q: q, re: c.headRe, miner: "time-current"},
			probe{q: q, re: c.oldRe, miner: "time-asof", asOf: c.asOf})
	}
	return out, nil
}

// timeProbeChains is the number of repaired chains mined into adversarial
// time-probe pairs (2 rows each) when n is large enough to afford them.
const timeProbeChains = 100

// mineProbes prints ~n TSV probes: adversarial time pairs from repaired
// chains, then ~1/3 supersession-chain probes, then paraphrase probes across
// all repos proportionally.
func MineProbes(db *sql.DB, n int) error {
	live, err := loadLiveFacts(db)
	if err != nil {
		return err
	}
	df := docFreq(live)
	nChains := timeProbeChains
	if n/5 < nChains { // small runs keep the old 1/3-chain, 2/3-paraphrase shape
		nChains = n / 5
	}
	timed, err := mineTimeProbes(db, df, live, nChains)
	if err != nil {
		return err
	}
	chain, err := mineChainProbes(db, df, live, (n-len(timed))/3)
	if err != nil {
		return err
	}
	para, err := mineParaphraseProbes(db, df, live, n-len(timed)-len(chain))
	if err != nil {
		return err
	}
	all := append(append(timed, chain...), para...)
	if len(all) == 0 {
		return fmt.Errorf("mined zero probes — DB too small or unenriched")
	}
	for _, p := range all {
		fmt.Printf("%s\t%s\t%s\t%s\n", p.q, p.re, p.miner, p.asOf)
	}
	fmt.Fprintf(os.Stderr, "mined %d probes (%d time-pair rows, %d chain, %d paraphrase)\n",
		len(all), len(timed), len(chain), len(para))
	return nil
}
