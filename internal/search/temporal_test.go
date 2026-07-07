package search

// Tests for temporal.go.

import (
	"oracle/internal/store"
	"testing"
	"time"
)

func TestClassifyTemporal(t *testing.T) {
	cases := []struct {
		q    string
		want TemporalIntent
	}{
		// CURRENT
		{"what is currently serving on atlas01", TemporalCurrent},
		{"deploy status of quasar", TemporalCurrent},
		{"what is the prod embedder", TemporalCurrent},
		{"which box hosts the API right now", TemporalCurrent},
		{"latest wheel version", TemporalCurrent},
		{"what are we shipping today", TemporalCurrent},
		{"are the workers healthy", TemporalCurrent},
		// HISTORICAL
		{"what was the prod embedder", TemporalHistorical},
		{"why did we move off atlas01", TemporalHistorical},
		{"topology as of 2026-06-01", TemporalHistorical},
		{"deploys in june", TemporalHistorical},
		{"history of the pgbouncer setup", TemporalHistorical},
		{"how did the serving layout originally look", TemporalHistorical},
		{"did the spot box get preempted", TemporalHistorical},
		{"what changed in the deploy script", TemporalHistorical},
		// historical cues beat current cues
		{"why did the status page break", TemporalHistorical},
		// NEUTRAL
		{"pgbouncer statement cache config", TemporalNeutral},
		{"atlas01 ssh access", TemporalNeutral},
		{"quasar deploy runbook", TemporalNeutral},
	}
	for _, c := range cases {
		if got := classifyTemporal(c.q); got != c.want {
			t.Errorf("classifyTemporal(%q) = %s, want %s", c.q, got, c.want)
		}
	}
}

func TestStaleStamps(t *testing.T) {
	now := float64(time.Now().Unix())
	day := 86400.0
	// status half-life 7d: stale past 14d, rotten past 21d
	if isStale("status", now-10*day, now) {
		t.Error("10d status should not be stale")
	}
	if !isStale("status", now-15*day, now) {
		t.Error("15d status should be stale")
	}
	if isRotten("status", now-20*day, now) {
		t.Error("20d status should not be rotten")
	}
	if !isRotten("status", now-22*day, now) {
		t.Error("22d status should be rotten")
	}
	// unknown kind defaults to 60d half-life
	if isStale("mystery", now-100*day, now) {
		t.Error("100d unknown-kind fact should not be stale (120d threshold)")
	}
	if !isStale("preference", now-800*day, now) {
		t.Error("800d preference should be stale (730d threshold)")
	}
	fo := FactOut{Stale: true, AsOfDate: "2026-01-02"}
	if StaleMark(fo) != "stale " {
		t.Errorf("staleMark stale = %q", StaleMark(fo))
	}
	if StaleMark(FactOut{}) != "" {
		t.Error("staleMark fresh should be empty")
	}
	if got := store.AsOfDate(1767312000); got != "2026-01-02" {
		t.Errorf("asOfDate = %q", got)
	}
}
