package ask

// ask: the kb-style LLM query engine, done properly — a frontier-LLM reasoner that
// plans multi-step retrieval over the graph with tools (search / entity /
// graph / metric), then answers with fact-id citations.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"oracle/internal/ingest"
	"oracle/internal/kb"
	"oracle/internal/llm"
	"oracle/internal/search"
	"oracle/internal/store"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const askSystem = `You are oracle, a reasoner over a bi-temporal fact graph extracted from all of the user's past AI coding sessions. Answer the question by INVESTIGATING with the tools — reformulate queries, follow entities, compare time periods, pull metric series — until you can answer precisely. Rules:
- Facts are the only ground truth. Cite fact ids inline like [123]. Never invent.
- Facts conflict across time: prefer newer (lower age_days); superseded history is visible via as_of.
- If coverage is missing, say exactly what's unknown.
- 2-6 tool calls is typical. Then answer tersely: conclusion first, caveats after.
- Time-stamp perishable claims: for any status/todo (or otherwise fast-decaying) fact you rely on, state its as-of date (the fact's as_of_date). If the fact is marked stale, say "as of <date>, ..." — never assert it in the present tense.`

// askLoopHints is appended to askSystem ONLY in the classic frontier-LLM loop.
// It must never reach the local synth model: that 1.5B was trained against
// askSystem verbatim, and these lines describe loop mechanics (seeding,
// batching) that don't exist in its one-shot synthesis call.
const askLoopHints = `
- The SEED message already contains hybrid-search results for the verbatim question. Do not re-run that exact search; go straight to the answer if they suffice, else refine the phrasing or pivot to entity/graph/metric. 0-4 further tool calls is typical.
- Batch independent lookups: emit them as multiple tool calls in ONE turn — they execute together. One call per turn is only for lookups that depend on a previous result.`

var askTools = []map[string]any{
	{"type": "function", "function": map[string]any{
		"name": "search", "description": "Hybrid (lexical+semantic) search over facts. Reformulate freely; try multiple phrasings.",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{
			"q":     map[string]any{"type": "string"},
			"repo":  map[string]any{"type": "string", "description": "optional repo boost"},
			"as_of": map[string]any{"type": "string", "description": "optional YYYY-MM-DD: state of knowledge then"},
			"k":     map[string]any{"type": "integer"},
		}, "required": []string{"q"}}}},
	{"type": "function", "function": map[string]any{
		"name": "entity", "description": "Everything known about one named thing: its facts + co-mentioned entities.",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{
			"name": map[string]any{"type": "string"},
		}, "required": []string{"name"}}}},
	{"type": "function", "function": map[string]any{
		"name": "graph", "description": "Typed relation traversal (S-P-O triples) from an entity, n hops.",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"hops": map[string]any{"type": "integer"},
		}, "required": []string{"name"}}}},
	{"type": "function", "function": map[string]any{
		"name": "metric", "description": "Numeric time series for a metric (snake_case), optionally scoped to an entity. Empty name lists available metrics.",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{
			"name":   map[string]any{"type": "string"},
			"entity": map[string]any{"type": "string"},
		}}}},
}

// ask runs the reasoner loop. Returns (answer, facts surfaced along the way,
// confidence). A below-threshold answer comes back already prefixed with the
// abstain marker (see confidence.go).
func ask(db *sql.DB, q, repo string, asOf float64) (string, []search.FactOut, AskConfidence, error) {
	if _, err := llm.Config(); err != nil {
		return "", nil, AskConfidence{}, err
	}
	userQ := q
	if repo != "" {
		userQ += fmt.Sprintf(" (focus repo: %s)", repo)
	}
	if asOf > 0 {
		userQ += " (answer as of the given historical date — pass as_of to search)"
	}
	system := askSystem + askLoopHints + "\n\n" + search.Introspect(db)
	if past := search.PastInvestigations(db, q); past != "" {
		system += "\n" + past
	}
	messages := []map[string]any{
		{"role": "system", "content": system},
		{"role": "user", "content": userQ},
	}
	seen := map[int64]search.FactOut{}
	var order []int64
	var callLog []string
	var searchTops []float64 // top-hit RRF score per search call (0 = no matches)

	// Seed round: the reasoner's first move is almost always search(<the
	// question>), which costs a full LLM round-trip just to ask for it. Run
	// that search before the loop and hand the results over — easy questions
	// finish in one LLM call, hard ones start already oriented.
	defer store.RecordLatency("ask_total", time.Now())
	seedHits, err := search.Search(db, q, repo, 8, asOf, false)
	if err != nil {
		return "", nil, AskConfidence{}, fmt.Errorf("seed search: %w", err)
	}
	top := 0.0
	if len(seedHits) > 0 {
		top = seedHits[0].Score
	}
	searchTops = append(searchTops, top)
	callLog = append(callLog, "search("+store.Truncate(q, 120)+") [seed]")
	for _, h := range seedHits {
		if _, ok := seen[h.ID]; !ok {
			seen[h.ID] = h
			order = append(order, h.ID)
		}
	}
	messages = append(messages, map[string]any{"role": "system",
		"content": "SEED — hybrid search results for the verbatim question:\n" + compactFacts(seedHits)})

	nudged := false
	const maxRounds = 8
	for round := 0; round < maxRounds; round++ {
		req := map[string]any{
			"messages":              messages,
			"tools":                 askTools,
			"max_completion_tokens": 16000,
			// no reasoning_effort here: some servers reject it alongside function tools
		}
		if round == maxRounds-1 {
			// budget exhausted: force a final answer instead of erroring out.
			// If the investigation came up empty this is where the reasoner
			// states what's missing; the confidence gate then abstains on it.
			req["tool_choice"] = "none"
			messages = append(messages, map[string]any{"role": "system",
				"content": "Tool budget exhausted. Answer now from the facts already surfaced, citing [ids]. If coverage is missing, say exactly what the graph does not contain."})
		}
		roundStart := time.Now()
		raw, err := llm.PostChat(req)
		if err != nil {
			return "", nil, AskConfidence{}, err
		}
		store.RecordLatency("ask_llm_round", roundStart)
		var cr struct {
			Choices []struct {
				Message struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(raw, &cr); err != nil {
			return "", nil, AskConfidence{}, err
		}
		if len(cr.Choices) == 0 {
			return "", nil, AskConfidence{}, fmt.Errorf("no choices")
		}
		msg := cr.Choices[0].Message
		if len(msg.ToolCalls) == 0 {
			if strings.TrimSpace(msg.Content) == "" {
				return "", nil, AskConfidence{}, fmt.Errorf("empty answer")
			}
			cited := citedIDs(msg.Content, seen)
			search.LogAskTrace(db, q, callLog, cited)
			if asOf == 0 { // historical questions are not usage signal
				search.ReinforceFacts(db, cited)
			}
			conf := askConfidence(msg.Content, cited, seen, searchTops)
			return applyAbstain(msg.Content, conf), collect(seen, order), conf, nil
		}
		// echo the assistant tool-call turn, then execute each call
		tcs := make([]map[string]any, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			tcs = append(tcs, map[string]any{"id": tc.ID, "type": "function",
				"function": map[string]any{"name": tc.Function.Name, "arguments": tc.Function.Arguments}})
		}
		messages = append(messages, map[string]any{"role": "assistant", "content": nil, "tool_calls": tcs})
		for _, tc := range msg.ToolCalls {
			callLog = append(callLog, tc.Function.Name+"("+store.Truncate(tc.Function.Arguments, 120)+")")
			result := runTool(db, tc.Function.Name, tc.Function.Arguments, repo, asOf, seen, &order, &searchTops)
			messages = append(messages, map[string]any{"role": "tool", "tool_call_id": tc.ID,
				"content": store.Truncate(result, 12000)})
		}
		// anti-muzero: once retrieval is strong and coverage is broad, tell
		// the reasoner so — a soft push toward answering over another
		// exploration round. Never forces (tool_choice stays free).
		if !nudged && retrievalStrength(searchTops) >= 0.8 && len(seen) >= 10 {
			nudged = true
			messages = append(messages, map[string]any{"role": "system",
				"content": "Retrieval is strong and coverage looks broad. Answer next turn unless a specific, nameable gap remains."})
		}
	}
	return "", nil, AskConfidence{}, fmt.Errorf("no answer after %d tool rounds despite forced finalization", maxRounds)
}

func runTool(db *sql.DB, name, argsJSON, defRepo string, defAsOf float64,
	seen map[int64]search.FactOut, order *[]int64, searchTops *[]float64) string {
	var a struct {
		Q, Repo, AsOf, Name, Entity string
		K, Hops                     int
	}
	_ = json.Unmarshal([]byte(argsJSON), &a)
	switch name {
	case "search":
		repo := a.Repo
		if repo == "" {
			repo = defRepo
		}
		asOf := defAsOf
		if a.AsOf != "" {
			if t, err := timeParseDate(a.AsOf); err == nil {
				asOf = t
			}
		}
		k := a.K
		if k <= 0 || k > 20 {
			k = 8
		}
		// reinforce=false: reasoner reformulations are not usage signal; the
		// facts the final answer cites are reinforced in ask() instead.
		hits, err := search.Search(db, a.Q, repo, k, asOf, false)
		if err != nil {
			return "error: " + err.Error()
		}
		top := 0.0
		if len(hits) > 0 {
			top = hits[0].Score
		}
		*searchTops = append(*searchTops, top)
		for _, h := range hits {
			if _, ok := seen[h.ID]; !ok {
				seen[h.ID] = h
				*order = append(*order, h.ID)
			}
		}
		return compactFacts(hits)
	case "entity":
		v, err := search.EntityView(db, a.Name, 15)
		if err != nil {
			return "error: " + err.Error()
		}
		if facts, ok := v["facts"].([]search.FactOut); ok {
			for _, h := range facts {
				if _, dup := seen[h.ID]; !dup {
					seen[h.ID] = h
					*order = append(*order, h.ID)
				}
			}
		}
		b, _ := json.Marshal(v)
		return string(b)
	case "graph":
		hops := a.Hops
		if hops <= 0 || hops > 3 {
			hops = 2
		}
		v, err := kb.Traverse(db, a.Name, hops, 40)
		if err != nil {
			return "error: " + err.Error()
		}
		b, _ := json.Marshal(v)
		return string(b)
	case "metric":
		if a.Name == "" {
			return ingest.MetricsList(db)
		}
		v, err := ingest.MetricSeries(db, a.Name, a.Entity)
		if err != nil {
			return "error: " + err.Error()
		}
		b, _ := json.Marshal(v)
		return string(b)
	}
	return "unknown tool " + name
}

var citeRe = regexp.MustCompile(`\[(\d+)\]`)

// citedIDs returns fact ids cited like [123] in the answer, in order of first
// appearance, restricted to facts the investigation actually surfaced.
func citedIDs(answer string, seen map[int64]search.FactOut) []int64 {
	var out []int64
	dup := map[int64]bool{}
	for _, m := range citeRe.FindAllStringSubmatch(answer, -1) {
		id, err := strconv.ParseInt(m[1], 10, 64)
		if err != nil || dup[id] {
			continue
		}
		if _, ok := seen[id]; !ok {
			continue
		}
		dup[id] = true
		out = append(out, id)
	}
	return out
}

func compactFacts(hits []search.FactOut) string {
	var b strings.Builder
	for _, h := range hits {
		fmt.Fprintf(&b, "[%d] (%s, %s, %.1fd, conf %.2f) %s\n", h.ID, h.Kind, h.Repo, h.AgeDays, h.Confidence, h.Statement)
	}
	if b.Len() == 0 {
		return "no matches"
	}
	return b.String()
}

func collect(seen map[int64]search.FactOut, order []int64) []search.FactOut {
	out := make([]search.FactOut, 0, len(order))
	for _, id := range order {
		out = append(out, seen[id])
	}
	if len(out) > 15 {
		out = out[:15]
	}
	return out
}

func timeParseDate(s string) (float64, error) {
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return 0, err
	}
	return float64(t.Unix()), nil
}
