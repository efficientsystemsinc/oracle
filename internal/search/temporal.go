package search

// Query-time temporal intent + staleness stamps.
//
// classifyTemporal is pure heuristics (no LLM): a query asking about the
// present ("what is X doing now") should rank fresh perishable facts up and
// demote rotten status/todo facts; a query asking about the past ("what was X
// in june") should not. NEUTRAL leaves ranking untouched.

import (
	"regexp"
)

type TemporalIntent int

const (
	TemporalNeutral TemporalIntent = iota
	TemporalCurrent
	TemporalHistorical
)

func (t TemporalIntent) String() string {
	switch t {
	case TemporalCurrent:
		return "current"
	case TemporalHistorical:
		return "historical"
	}
	return "neutral"
}

// perishableKinds: kinds whose truth rots fast enough that a CURRENT query
// should actively prefer freshness (matches the short half-lives above).
var perishableKinds = map[string]bool{"status": true, "todo": true}

// Historical wins over current when both cue sets appear ("why did X ... now"?
// rare, but a past-tense/why-did framing is asking for history, not state).
var (
	histPhraseRe = regexp.MustCompile(`(?i)\b(as of|why did|how did|used to|back (then|when|in)|at the time|in (january|february|march|april|may|june|july|august|september|october|november|december|20\d\d))\b`)
	histTokenRe  = regexp.MustCompile(`(?i)\b(was|were|did|had|history|historically|originally|previously|before|earlier|changed|evolved|timeline)\b`)
	currPhraseRe = regexp.MustCompile(`(?i)\b(right now|as of (now|today)|at the moment|these days)\b`)
	currTokenRe  = regexp.MustCompile(`(?i)\b(currently|now|today|latest|current|status|still|anymore|nowadays)\b`)
	// bare present-tense copulas count as a current cue only weakly — they are
	// everywhere ("what is the schema") — but per spec they signal CURRENT when
	// no historical cue is present.
	presentTokenRe = regexp.MustCompile(`(?i)\b(is|are)\b`)
)

// classifyTemporal decides the query's time frame. Precedence:
// explicit historical cues > explicit current cues > bare present tense > neutral.
func classifyTemporal(q string) TemporalIntent {
	if histPhraseRe.MatchString(q) || histTokenRe.MatchString(q) {
		return TemporalHistorical
	}
	if currPhraseRe.MatchString(q) || currTokenRe.MatchString(q) || presentTokenRe.MatchString(q) {
		return TemporalCurrent
	}
	return TemporalNeutral
}

// staleAfter: a fact is stamped stale when its age exceeds 2x its kind's
// half-life; rottenAfter (3x) is the CURRENT-query hard-demotion horizon.
func kindHalfLifeSecs(kind string) float64 {
	hl := halfLifeDays[kind]
	if hl == 0 {
		hl = 60
	}
	return hl * 86400
}

func isStale(kind string, validFrom, ref float64) bool {
	return ref-validFrom > 2*kindHalfLifeSecs(kind)
}

func isRotten(kind string, validFrom, ref float64) bool {
	return ref-validFrom > 3*kindHalfLifeSecs(kind)
}

// staleMark renders the CLI stamp: stale facts get a loud "stale " prefix;
// the as-of date is always printed alongside (see main.go query/ask output).
func StaleMark(f FactOut) string {
	if f.Stale {
		return "stale "
	}
	return ""
}
