package ask

// Confidence + abstain for ask. All features are computed post-hoc from the
// investigation the reasoner already ran — no extra LLM call.
//
// Formula (simple, monotone; weights hand-set, THRESHOLD fit on the build
// split of eval/ask_confidence probes and reported on the holdout — see
// eval/ask_confidence.md):
//
//	cov  = min(#distinct-valid-cited / 3, 1)           coverage
//	val  = valid-citations / total-citation-tokens      citation validity
//	tier = mean over cited of 0.5*conf + 0.5*evTier     fact quality
//	       evTier: verified 1.0, reported 0.8, else 0.6
//	agr  = frac of cited with NO live contradicts edge  agreement
//	ret  = mean over searches of min(topScore/0.03, 1)  retrieval strength
//
//	base = 0.30*cov + 0.25*val + 0.20*tier + 0.15*agr + 0.10*ret
//	confidence = base * 0.4 if the answer LEADS with absence/unknown
//	             language (leadsWithDoubt), else base
//
// Every term is 0 when nothing is cited except ret, so an uncited answer
// lands at <= 0.10 and abstains. The doubt damp exists because a grounded
// negative ("we don't use X; we use Y [ids]") cites real facts and would
// otherwise score ~0.9 while the reasoner itself is reporting no coverage. confidence < askAbstainThreshold => the
// answer is prefixed with the abstain marker.

import (
	"math"
	"oracle/internal/search"
	"strings"
)

// askAbstainThreshold: fit on the build split (2026-07-05, see
// eval/ask_confidence.md). With the doubt damp, build separation was total
// (answerable min 0.62 vs damped-unanswerable max 0.38; sweep flat across
// 0.40-0.50); 0.45 sits mid-gap.
const AskAbstainThreshold = 0.45

const abstainPrefix = "LOW CONFIDENCE — likely not covered:"

type AskFeatures struct {
	Coverage         int     `json:"coverage"`          // distinct valid cited facts
	CitationValidity float64 `json:"citation_validity"` // valid ids / citation tokens
	MeanTier         float64 `json:"mean_tier"`         // blended confidence+evidence tier of cited
	Agreement        float64 `json:"agreement"`         // frac cited without live contradicts
	Retrieval        float64 `json:"retrieval"`         // mean normalized top RRF score per search
}

type AskConfidence struct {
	Score     float64     `json:"score"`
	Abstained bool        `json:"abstained"`
	Features  AskFeatures `json:"features"`
}

func evidenceTier(ev string) float64 {
	switch ev {
	case "verified":
		return 1.0
	case "reported":
		return 0.8
	default: // asserted, or unset (entity/graph-sourced hits)
		return 0.6
	}
}

// askConfidence scores the final answer. cited are the valid cited ids (from
// citedIDs), totalCites counts every [N] token in the answer (valid or not),
// searchTops holds the top-hit RRF score of each search the reasoner ran
// (0 for a search that returned nothing).
func askConfidence(answer string, cited []int64, seen map[int64]search.FactOut, searchTops []float64) AskConfidence {
	totalCites := len(citeRe.FindAllString(answer, -1))

	cov := math.Min(float64(len(cited))/3.0, 1)

	val := 0.0
	if totalCites > 0 {
		val = float64(len(cited)) / float64(totalCites)
	}

	tier, agr := 0.0, 0.0
	if len(cited) > 0 {
		agreeing := 0
		for _, id := range cited {
			f := seen[id]
			tier += 0.5*f.Confidence + 0.5*evidenceTier(f.Evidence)
			if len(f.ContraBy) == 0 {
				agreeing++
			}
		}
		tier /= float64(len(cited))
		agr = float64(agreeing) / float64(len(cited))
	}

	ret := 0.0
	if len(searchTops) > 0 {
		for _, s := range searchTops {
			ret += math.Min(s/0.03, 1)
		}
		ret /= float64(len(searchTops))
	}

	score := 0.30*cov + 0.25*val + 0.20*tier + 0.15*agr + 0.10*ret
	// self-report damp: if the answer LEADS with absence/unknown language
	// ("we don't use X", "unknown - no fact states..."), the reasoner itself is
	// saying the graph lacks coverage - citations of adjacent facts must not
	// mask that. Lead-only: mid-answer caveats on a confident answer don't damp.
	if leadsWithDoubt(answer) {
		score *= 0.4
	}
	return AskConfidence{
		Score:     math.Round(score*1000) / 1000,
		Abstained: score < AskAbstainThreshold,
		Features: AskFeatures{
			Coverage:         len(cited),
			CitationValidity: math.Round(val*1000) / 1000,
			MeanTier:         math.Round(tier*1000) / 1000,
			Agreement:        math.Round(agr*1000) / 1000,
			Retrieval:        math.Round(ret*1000) / 1000,
		},
	}
}

// applyAbstain enforces the abstain contract: a below-threshold answer must
// LEAD with the low-confidence marker. The reasoner's own text already states
// what's missing (its system prompt requires it), so no second LLM call.
func applyAbstain(answer string, conf AskConfidence) string {
	if !conf.Abstained {
		return answer
	}
	if strings.HasPrefix(strings.TrimSpace(answer), abstainPrefix) {
		return answer
	}
	return abstainPrefix + " " + strings.TrimSpace(answer)
}

// doubtMarkers: explicit absence/unknown phrasings. Shared by the confidence
// damp (lead-only) and askeval's grounded-negative classifier (whole answer).
var doubtMarkers = []string{
	"unknown", "don't have", "do not have", "don't use", "do not use",
	"doesn't use", "don't appear", "does not appear", "not used", "no evidence",
	"not covered", "no fact", "no cited fact", "nothing in the", "no indication",
	"not in the graph", "graph does not contain", "no record", "no recorded",
	"not present", "not mentioned", "no mention", "did not have", "didn't have",
	"no decision", "found no", "couldn't find", "could not find", "nothing recorded",
	"not configured", "isn't configured", "can't verify", "cannot verify",
	"can't confirm", "cannot confirm", "can't find", "cannot find", "unable to find",
	"not set up", "don't run", "do not run", "not have", "don't find",
	"do not find", "no documented", "not documented", "not recorded",
}

// leadsWithDoubt reports whether the first ~160 chars of the answer contain an
// absence/unknown marker (curly apostrophes normalized).
func leadsWithDoubt(answer string) bool {
	lead := strings.ToLower(strings.ReplaceAll(answer, "\u2019", "'"))
	lead = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(lead), strings.ToLower(abstainPrefix)))
	if len(lead) > 160 {
		lead = lead[:160]
	}
	for _, m := range doubtMarkers {
		if strings.Contains(lead, m) {
			return true
		}
	}
	return false
}
