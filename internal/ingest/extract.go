package ingest

// LLM extraction: transcript chunk -> durable facts, via any
// OpenAI-compatible endpoint (see provider.go).
// Fails loudly — no fallback provider, no empty-list-on-error.

import (
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"oracle/internal/truth"
	"strings"
	"time"
)

const (
	systemPrompt = `You extract durable memory from AI coding-agent session transcripts for a team knowledge graph called oracle.

Extract only facts worth remembering NEXT WEEK: decisions made (and why), hard-won gotchas, infrastructure/config truths, benchmark numbers, user preferences/workflow rules, project status changes, and standing TODOs. Skip: transient debugging noise, file-by-file edit narration, anything true only for this one session, generic knowledge.

Each fact must be a standalone atomic statement understandable with zero session context — name the systems, repos, boxes, numbers explicitly. Convert relative dates to absolute when the session date is given.

For each fact include "quote": the shortest verbatim transcript excerpt (<=200 chars) that supports it — the evidence span, copied exactly.
For each fact set "evidence": "verified" if the transcript SHOWS confirming output (command results, test output, API responses), "asserted" if an agent/user merely states it, "reported" if it is secondhand ("the docs say", "X told me").
Return JSON: {"facts":[{"statement":str, "kind":"decision|fact|gotcha|preference|status|todo", "entities":[str, canonical short names of systems/repos/boxes/tools mentioned], "files":[repo-relative paths if central], "confidence":0.0-1.0, "evidence":"verified|asserted|reported", "quote":str}]}
Return {"facts":[]} if nothing durable. Max 12 facts per chunk; prefer fewer, denser.`

	judgePrompt = `You maintain a fact graph. For each NEW fact below, decide if it SUPERSEDES (replaces/outdates) any of the listed OLD facts — same topic, newer state. Duplicate = same claim -> supersede. Different aspects of the same system -> no.
Also flag CONTRADICTIONS: old facts that CONFLICT with the new one where both claim to be current (not a clean replacement).
Each fact carries [valid_from YYYY-MM-DD, evidence verified|asserted|reported]. Rules:
- A fact may only supersede an OLDER fact: the NEW fact's valid_from must be later than or equal to the OLD fact's (newer valid_from wins). Never let an older-dated fact supersede a newer-dated one.
- An 'asserted' fact must NOT supersede a 'verified' fact unless it EXPLICITLY reports the change (e.g. "X moved / changed / is now Y").
- When unsure, prefer contradicts over supersedes.
Return JSON: {"verdicts":[{"new_idx":int, "supersedes":[old_id,...], "contradicts":[old_id,...]}]} with an entry only when either list is non-empty.`
)

var validKinds = map[string]bool{
	"decision": true, "fact": true, "gotcha": true,
	"preference": true, "status": true, "todo": true,
}

type Fact struct {
	Statement  string   `json:"statement"`
	Kind       string   `json:"kind"`
	Entities   []string `json:"entities"`
	Files      []string `json:"files"`
	Confidence float64  `json:"confidence"`
	Evidence   string   `json:"evidence"`
	Quote      string   `json:"quote"`
}

func ExtractFacts(chunkText, repo, sessionDate string) ([]Fact, error) {
	user := fmt.Sprintf("Repo: %s\nSession date: %s\n\nTRANSCRIPT:\n%s", repo, sessionDate, chunkText)
	var out struct {
		Facts []Fact `json:"facts"`
	}
	call := llm.ChatJSON
	if llm.LocalExtractEnabled() {
		call = llm.ChatJSONLocal
	}
	if err := call(systemPrompt, user, 28000, &out); err != nil {
		return nil, err
	}
	var clean []Fact
	for _, f := range out.Facts {
		f.Statement = redact(strings.TrimSpace(f.Statement)) // defense in depth: statements are LLM output
		if f.Statement == "" {
			continue
		}
		if !validKinds[f.Kind] {
			f.Kind = "fact"
		}
		if f.Confidence <= 0 || f.Confidence > 1 {
			f.Confidence = 0.7
		}
		switch f.Evidence {
		case "verified": // transcript-confirmed: trust as extracted
		case "reported": // secondhand: cap hard until corroborated
			if f.Confidence > 0.6 {
				f.Confidence = 0.6
			}
		default: // bare assertion: single-source cap (anti over-index)
			f.Evidence = "asserted"
			if f.Confidence > 0.8 {
				f.Confidence = 0.8
			}
		}
		if len(f.Entities) > 12 {
			f.Entities = f.Entities[:12]
		}
		if len(f.Files) > 12 {
			f.Files = f.Files[:12]
		}
		clean = append(clean, f)
	}
	return clean, nil
}

type OldFact struct {
	ID        int64
	Statement string
	ValidFrom float64 // unix seconds, world time
	Evidence  string  // verified|asserted|reported
}

// judgeUserPrompt renders the judge's user message. newDate is the NEW facts'
// valid_from (session date, YYYY-MM-DD). Split out for testability.
func JudgeUserPrompt(newFacts []Fact, candidates map[int][]OldFact, newDate string) string {
	var b strings.Builder
	for idx, olds := range candidates {
		f := newFacts[idx]
		fmt.Fprintf(&b, "NEW %d [valid_from %s, evidence %s]: %s\n", idx, newDate, f.Evidence, f.Statement)
		for _, o := range olds {
			fmt.Fprintf(&b, "  OLD %d [valid_from %s, evidence %s]: %s\n",
				o.ID, store.AsOfDate(o.ValidFrom), o.Evidence, o.Statement)
		}
	}
	return b.String()
}

// judgeSupersede: candidates[newIdx] = plausible old facts. -> newIdx -> old ids.
// newDate = the NEW facts' valid_from date (YYYY-MM-DD, session date).
// db is used only for judge_shadow logging when ORACLE_LOCAL_JUDGE != off.
func JudgeSupersede(db truth.DBE, newFacts []Fact, candidates map[int][]OldFact, newDate string) (map[int][]int64, map[int][]int64, error) {
	if len(candidates) == 0 {
		return nil, nil, nil
	}
	mode, err := truth.LocalJudgeMode()
	if err != nil {
		return nil, nil, err
	}
	if mode != truth.LocalJudgeOff {
		if err := truth.EnsureJudgeShadow(db); err != nil {
			return nil, nil, err
		}
	}

	// ACTIVE: high-margin pairs take the local verdict without the frontier LLM; only
	// the residual (below-threshold) pairs batch to the LLM exactly as today.
	// Ingest judges BEFORE the new fact is inserted, so new_id logs as NULL.
	localSup, localCon := map[int][]int64{}, map[int][]int64{}
	if mode == truth.LocalJudgeActive {
		threshold, err := truth.LocalJudgeMargin()
		if err != nil {
			return nil, nil, err
		}
		now := float64(time.Now().Unix())
		residual := map[int][]OldFact{}
		for idx, olds := range candidates {
			f := newFacts[idx]
			newSide := truth.RenderJudgeSide(f.Statement, newDate, f.Evidence, f.Kind)
			for _, o := range olds {
				oldSide := truth.RenderJudgeSide(o.Statement, store.AsOfDate(o.ValidFrom), o.Evidence, f.Kind)
				v, ok, err := truth.ActiveJudgePair(db, now, o.ID, 0, threshold, oldSide, newSide)
				if err != nil {
					return nil, nil, err
				}
				if !ok {
					residual[idx] = append(residual[idx], o)
					continue
				}
				switch v {
				case "supersedes":
					localSup[idx] = append(localSup[idx], o.ID)
				case "contradicts":
					localCon[idx] = append(localCon[idx], o.ID)
				} // "nothing": decided locally, no edge
			}
		}
		candidates = residual
		if len(candidates) == 0 {
			return localSup, localCon, nil
		}
	}

	var out struct {
		Verdicts []struct {
			NewIdx      int     `json:"new_idx"`
			Supersedes  []int64 `json:"supersedes"`
			Contradicts []int64 `json:"contradicts"`
		} `json:"verdicts"`
	}
	if err := llm.ChatJSON(judgePrompt, JudgeUserPrompt(newFacts, candidates, newDate), 8000, &out); err != nil {
		return nil, nil, err
	}
	sup, con := localSup, localCon
	for _, v := range out.Verdicts {
		olds, ok := candidates[v.NewIdx]
		if !ok {
			continue
		}
		allowed := map[int64]bool{}
		for _, o := range olds {
			allowed[o.ID] = true
		}
		for _, id := range v.Supersedes {
			if allowed[id] {
				sup[v.NewIdx] = append(sup[v.NewIdx], id)
			}
		}
		for _, id := range v.Contradicts {
			if allowed[id] {
				con[v.NewIdx] = append(con[v.NewIdx], id)
			}
		}
	}

	// SHADOW: run the local judge on every pair the frontier LLM just decided and log
	// agreement. Zero behavior change: shadowJudgePair never returns an error.
	if mode == truth.LocalJudgeShadow {
		now := float64(time.Now().Unix())
		for idx, olds := range candidates {
			f := newFacts[idx]
			newSide := truth.RenderJudgeSide(f.Statement, newDate, f.Evidence, f.Kind)
			inSup, inCon := map[int64]bool{}, map[int64]bool{}
			for _, id := range sup[idx] {
				inSup[id] = true
			}
			for _, id := range con[idx] {
				inCon[id] = true
			}
			for _, o := range olds {
				llm := "nothing"
				switch {
				case inSup[o.ID]:
					llm = "supersedes"
				case inCon[o.ID]:
					llm = "contradicts"
				}
				oldSide := truth.RenderJudgeSide(o.Statement, store.AsOfDate(o.ValidFrom), o.Evidence, f.Kind)
				truth.ShadowJudgePair(db, now, o.ID, 0, llm, oldSide, newSide)
			}
		}
	}
	return sup, con, nil
}
