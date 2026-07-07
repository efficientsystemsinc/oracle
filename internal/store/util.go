package store

import (
	"os"
	"time"
)

func Truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// asOfDate renders a unix timestamp as YYYY-MM-DD in UTC — the one shared
// date renderer for LLM prompts and audit output.
func AsOfDate(validFrom float64) string {
	return time.Unix(int64(validFrom), 0).UTC().Format("2006-01-02")
}

// localDate renders a unix timestamp as YYYY-MM-DD in local time, for
// human-facing CLI output.
func LocalDate(ts float64) string { return time.Unix(int64(ts), 0).Format("2006-01-02") }

// localEmbedEnabled is a var (reading the env each call) so tests can toggle
// via t.Setenv.
func LocalEmbedEnabled() bool {
	return os.Getenv("ORACLE_LOCAL_EMBED") == "1"
}
