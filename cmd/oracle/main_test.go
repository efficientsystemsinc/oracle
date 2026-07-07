package main

// Tests for main.go.

import (
	"flag"
	"strings"
	"testing"
)

func TestSystemdUnit(t *testing.T) {
	u := systemdUnit("/usr/local/bin/oracle", "/home/sam/.oracle/daemon.log")
	for _, want := range []string{
		"ExecStart=/usr/local/bin/oracle up",
		"Restart=always",
		"StandardOutput=append:/home/sam/.oracle/daemon.log",
		"WantedBy=default.target",
	} {
		if !strings.Contains(u, want) {
			t.Errorf("unit missing %q:\n%s", want, u)
		}
	}
	for _, section := range []string{"[Unit]", "[Service]", "[Install]"} {
		if strings.Count(u, section) != 1 {
			t.Errorf("unit must have exactly one %s section", section)
		}
	}
}

func TestLaunchdPlist(t *testing.T) {
	p := launchdPlist("/opt/homebrew/bin/oracle", "/Users/sam/.oracle/daemon.log")
	for _, want := range []string{
		"<string>/opt/homebrew/bin/oracle</string><string>up</string>",
		"<key>KeepAlive</key><true/>",
		"<string>/Users/sam/.oracle/daemon.log</string>",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("plist missing %q", want)
		}
	}
}

func TestParseWithPositional(t *testing.T) {
	mk := func() (*flag.FlagSet, *string, *int, *bool) {
		fs := flag.NewFlagSet("t", flag.ContinueOnError)
		asOf := fs.String("as-of", "", "")
		k := fs.Int("k", 10, "")
		js := fs.Bool("json", false, "")
		return fs, asOf, k, js
	}
	// flag with value BEFORE the positional — the old splitter stole the value
	fs, asOf, _, _ := mk()
	if pos := parseWithPositional(fs, []string{"--as-of", "2026-06-20", "prod topology"}); pos != "prod topology" {
		t.Fatalf("positional = %q", pos)
	}
	if *asOf != "2026-06-20" {
		t.Fatalf("as-of = %q", *asOf)
	}
	// flags AFTER the positional
	fs, _, k, _ := mk()
	if pos := parseWithPositional(fs, []string{"some text", "-k", "5"}); pos != "some text" || *k != 5 {
		t.Fatalf("pos=%q k=%d", pos, *k)
	}
	// boolean flag directly before the positional must not eat it
	fs, _, _, js := mk()
	if pos := parseWithPositional(fs, []string{"-json", "text"}); pos != "text" || !*js {
		t.Fatalf("pos=%q json=%v", pos, *js)
	}
	// no positional at all
	fs, _, _, _ = mk()
	if pos := parseWithPositional(fs, []string{"-k", "3"}); pos != "" {
		t.Fatalf("want empty, got %q", pos)
	}
}

func TestCommandRegistry(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range commands {
		if c.ns != "" && c.ns != "graph" && c.ns != "admin" {
			t.Errorf("command %q has unknown namespace %q", c.name, c.ns)
		}
		if seen[c.name] {
			t.Errorf("duplicate command short name %q", c.name)
		}
		seen[c.name] = true
		if c.summary == "" || c.usage == "" || c.run == nil {
			t.Errorf("command %q is missing summary/usage/run", c.name)
		}
	}
	for old, a := range legacyAliases {
		if findCommand(a[0], a[1]) == nil {
			t.Errorf("legacy alias %q points at missing command %q %q", old, a[0], a[1])
		}
	}
	// every core command must be reachable by findCommand("", name)
	for _, name := range []string{"query", "ask", "install", "up", "models", "status"} {
		if findCommand("", name) == nil {
			t.Errorf("core command %q missing", name)
		}
	}
}
