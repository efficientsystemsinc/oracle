package ask

// askLocal: the MuZero-style READ loop, productized — and fully local. A
// BC-trained policy model (Qwen2.5-0.5B, :8398) emits the tool sequence in
// one local call; we execute it with the same runTool machinery as classic
// ask; then a trained synthesis model (Qwen2.5-1.5B, :8397) writes the cited
// answer. ZERO frontier calls in the loop. Classic ask stays reachable via
// ORACLE_ASK_REMOTE=1 as the A/B escape hatch. Both models serve through any
// OpenAI-compatible server — scripts/ask_servers.sh picks mlx_lm on macOS,
// vllm/llama.cpp elsewhere.
//
// MCTS-lite widening: if the executed sequence retrieved nothing useful
// (retrieval-strength feature below its answerable level), the policy is
// asked for ONE alternate sequence and that is executed too. Confidence +
// abstain are computed by the exact same askConfidence as classic — the
// features are model-agnostic.
//
// No silent fallback: if either local server is down, askLocal errors loudly.
// It never quietly reroutes to the frontier LLM loop.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/search"
	"oracle/internal/store"
	"os"
	"regexp"
	"sort"
	"strings"
)

// policySystem is the exact prompt the policy was BC-trained with.
const policySystem = `You plan retrieval over a fact graph. Given a question, emit the tool sequence, one per line: search(q) | entity(name) | graph(name) | metric(name) | STOP.`

const (
	localMaxSteps = 4
	localMaxFacts = 15
	// localWeakRetrieval: widening trigger. The retrieval feature in
	// askConfidence is mean(min(topRRF/0.03, 1)) per search; on the
	// ask_confidence probe set answerable questions sit near 1.0 and
	// unanswerable near 0. Below 0.5 (or nothing surfaced at all) the first
	// sequence found nothing useful — ask the policy for one alternate.
	localWeakRetrieval = 0.5
)

func policyURL() string {
	if v := os.Getenv("ORACLE_POLICY_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "http://127.0.0.1:8398"
}

func synthURL() string {
	if v := os.Getenv("ORACLE_SYNTH_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "http://127.0.0.1:8397"
}

// useLocalAsk picks the ask backend. The 2026-07-07 askab gate FAILED on
// answer quality (local 55.0%% vs classic 77.5%% regex-hit; needed >=85%% of
// classic), so local is opt-in via ORACLE_ASK_LOCAL=1 until the policy/synth
// models improve. It wins on latency (p50 6.9s vs 19.0s) and runs zero
// frontier calls — rerun `oracle askab` after any model retrain.
func useLocalAsk() bool {
	return os.Getenv("ORACLE_ASK_LOCAL") == "1"
}

// askAuto dispatches to the configured ask backend.
func AskAuto(db *sql.DB, q, repo string, asOf float64) (string, []search.FactOut, AskConfidence, error) {
	if useLocalAsk() {
		return askLocal(db, q, repo, asOf)
	}
	return ask(db, q, repo, asOf)
}

type policyStep struct {
	tool string // search | entity | graph | metric
	arg  string
}

// policyLineRe: one tool call per line, e.g. `search(meadow ssh access)`.
var policyLineRe = regexp.MustCompile(`^(search|entity|graph|metric)\((.*)\)\s*$`)

// parsePolicyPlan parses the policy model's raw output into tool steps.
// Tolerant by design (the 0.5B model drifts): accepts STOP and STOP(),
// ignores duplicate STOPs and unparseable lines, dedupes repeated calls,
// caps at localMaxSteps.
func parsePolicyPlan(raw string) []policyStep {
	var steps []policyStep
	seen := map[string]bool{}
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		up := strings.ToUpper(strings.TrimSuffix(strings.TrimSuffix(line, "()"), "."))
		if up == "STOP" {
			break
		}
		m := policyLineRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		arg := strings.TrimSpace(strings.Trim(strings.TrimSpace(m[2]), `"'`))
		if arg == "" {
			continue
		}
		key := m[1] + "\x00" + strings.ToLower(arg)
		if seen[key] {
			continue
		}
		seen[key] = true
		steps = append(steps, policyStep{tool: m[1], arg: arg})
		if len(steps) == localMaxSteps {
			break
		}
	}
	return steps
}

// localChat: one call to an OpenAI-compatible local server (mlx_lm on macOS,
// vllm/llama.cpp elsewhere — scripts/ask_servers.sh serves all three).
// ORACLE_ASK_MODEL_FIELD sets the OpenAI "model" request field for servers
// that require one (vLLM; ask_servers.sh serves both roles as "oracle-ask").
// Unset = omitted, which is what mlx_lm and llama-server expect.
func localChat(baseURL, role, system, user string, maxTokens int) (string, error) {
	req := map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  maxTokens,
		"temperature": 0.0,
	}
	if m := os.Getenv("ORACLE_ASK_MODEL_FIELD"); m != "" {
		req["model"] = m
	}
	body, _ := json.Marshal(req)
	raw, err := llm.PostJSON(baseURL+"/v1/chat/completions", body, "")
	if err != nil {
		return "", fmt.Errorf("%s server (%s) unreachable — start it with scripts/ask_servers.sh: %w", role, baseURL, err)
	}
	var cr struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &cr); err != nil {
		return "", fmt.Errorf("%s server bad response: %w", role, err)
	}
	if len(cr.Choices) == 0 || strings.TrimSpace(cr.Choices[0].Message.Content) == "" {
		return "", fmt.Errorf("%s server returned empty completion", role)
	}
	return cr.Choices[0].Message.Content, nil
}

// policyPlan asks the local policy server for a tool sequence. note, when
// non-empty, conditions the retry ("previous attempt found nothing useful").
func policyPlan(question, note string) ([]policyStep, error) {
	user := question
	if note != "" {
		user += "\n(" + note + ")"
	}
	out, err := localChat(policyURL(), "policy", policySystem, user, 120)
	if err != nil {
		return nil, err
	}
	steps := parsePolicyPlan(out)
	if len(steps) == 0 {
		return nil, fmt.Errorf("policy emitted no parseable tool calls: %q", store.Truncate(out, 200))
	}
	return steps, nil
}

// stepArgs renders a policy step as the JSON args runTool expects.
func stepArgs(s policyStep) string {
	var m map[string]any
	switch s.tool {
	case "search":
		m = map[string]any{"q": s.arg}
	default: // entity, graph, metric all take a name
		m = map[string]any{"name": s.arg}
	}
	b, _ := json.Marshal(m)
	return string(b)
}

// retrievalStrength mirrors the ret feature of askConfidence over the
// searches executed so far.
func retrievalStrength(searchTops []float64) float64 {
	if len(searchTops) == 0 {
		return 0
	}
	ret := 0.0
	for _, s := range searchTops {
		if v := s / 0.03; v < 1 {
			ret += v
		} else {
			ret++
		}
	}
	return ret / float64(len(searchTops))
}

// askLocal is the local counterpart of ask: policy plans, tools run, the
// synth model answers. Same return contract as ask.
func askLocal(db *sql.DB, q, repo string, asOf float64) (string, []search.FactOut, AskConfidence, error) {
	seen := map[int64]search.FactOut{}
	var order []int64
	var callLog []string
	var searchTops []float64

	exec := func(steps []policyStep) {
		for _, s := range steps {
			callLog = append(callLog, s.tool+"("+store.Truncate(s.arg, 120)+")")
			runTool(db, s.tool, stepArgs(s), repo, asOf, seen, &order, &searchTops)
		}
	}

	steps, err := policyPlan(q, "")
	if err != nil {
		return "", nil, AskConfidence{}, err
	}
	exec(steps)

	// MCTS-lite widening: one alternate sequence if retrieval came up weak.
	if len(seen) == 0 || retrievalStrength(searchTops) < localWeakRetrieval {
		if alt, err := policyPlan(q, "previous attempt found nothing useful; try a different tool sequence"); err == nil {
			exec(alt)
		}
		// a failed retry is not fatal: synthesis states what's missing and
		// the confidence gate abstains.
	}

	// facts for synthesis: cap at localMaxFacts by retrieval score.
	facts := make([]search.FactOut, 0, len(order))
	for _, id := range order {
		facts = append(facts, seen[id])
	}
	sort.SliceStable(facts, func(i, j int) bool { return facts[i].Score > facts[j].Score })
	if len(facts) > localMaxFacts {
		facts = facts[:localMaxFacts]
	}

	// ONE local synthesis call: question + facts -> cited answer. The synth
	// model was trained on the ask conventions (newer-facts-win, as-of dating
	// of perishables, terse conclusion-first, [id] citations).
	user := q
	if repo != "" {
		user += fmt.Sprintf(" (focus repo: %s)", repo)
	}
	answer, err := localChat(synthURL(), "synth", askSystem, user+"\n\nFACTS:\n"+compactFacts(facts), 2000)
	if err != nil {
		return "", nil, AskConfidence{}, err
	}

	cited := citedIDs(answer, seen)
	search.LogAskTrace(db, q, callLog, cited)
	if asOf == 0 {
		search.ReinforceFacts(db, cited)
	}
	conf := askConfidence(answer, cited, seen, searchTops)
	// sources shown to the user = the synthesis input, same 15-cap as ask
	return applyAbstain(answer, conf), facts, conf, nil
}
