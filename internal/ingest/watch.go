package ingest

// Session watchers: tail Claude Code + codex jsonl files from stored byte
// offsets, render new content into plain-text transcript chunks.

import (
	"bufio"
	"encoding/json"
	"oracle/internal/store"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	chunkChars   = 24_000
	overlapChars = 2_000
	maxToolChars = 400
)

// overlapTail keeps whole trailing pieces up to overlapChars, so a fact whose
// evidence spans a chunk cut appears intact at the start of the next chunk.
// A single oversized trailing piece yields no overlap rather than a huge one.
// Returns the tail and its total length (the next chunk's starting acc).
func overlapTail(pieces []string) ([]string, int) {
	total, i := 0, len(pieces)
	for i > 0 && total+len(pieces[i-1]) <= overlapChars {
		total += len(pieces[i-1])
		i--
	}
	return append([]string(nil), pieces[i:]...), total
}

// ponytail: bench sandboxes under /private/tmp are huge and low-signal; skip
var skipProjectPrefixes = []string{"-private-tmp", "-tmp-", "-var-"}

var secretRe = regexp.MustCompile(
	`sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|xox[bap]-[A-Za-z0-9-]{10,}` +
		`|ghp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{40,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}` +
		`|(?i:(password|api[_-]?key|secret|token)\s*[=:]\s*)[^\s'"]{8,}`)

func redact(s string) string { return secretRe.ReplaceAllString(s, "[REDACTED]") }

type Chunk struct {
	Path      string
	Source    string
	Repo      string
	Session   string
	Text      string
	EndOffset int64
	EventTime float64
}

type SessionFile struct {
	Path   string
	Source string
}

func ClaudeDir() string { h, _ := os.UserHomeDir(); return filepath.Join(h, ".claude", "projects") }
func CodexDir() string  { h, _ := os.UserHomeDir(); return filepath.Join(h, ".codex", "sessions") }

func Discover(sinceDays float64) []SessionFile {
	var out []SessionFile
	var cutoff time.Time
	if sinceDays > 0 {
		cutoff = time.Now().Add(-time.Duration(sinceDays * 24 * float64(time.Hour)))
	}
	dirs, _ := os.ReadDir(ClaudeDir())
	for _, d := range dirs {
		if !d.IsDir() || hasAnyPrefix(d.Name(), skipProjectPrefixes) {
			continue
		}
		files, _ := filepath.Glob(filepath.Join(ClaudeDir(), d.Name(), "*.jsonl"))
		for _, f := range files {
			if fresh(f, cutoff) {
				out = append(out, SessionFile{f, "claude"})
			}
		}
	}
	_ = filepath.WalkDir(CodexDir(), func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(p, ".jsonl") && fresh(p, cutoff) {
			out = append(out, SessionFile{p, "codex"})
		}
		return nil
	})
	return out
}

func hasAnyPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

func fresh(path string, cutoff time.Time) bool {
	if cutoff.IsZero() {
		return true
	}
	st, err := os.Stat(path)
	return err == nil && st.ModTime().After(cutoff)
}

// repoCache memoizes cwd -> repo label; sessions repeat cwds constantly and
// resolution stats the filesystem. Concurrent-safe (readNew runs in parallel).
var repoCache sync.Map

func RepoFromCwd(cwd string) string {
	if cwd == "" {
		return "unknown"
	}
	if v, ok := repoCache.Load(cwd); ok {
		return v.(string)
	}
	label := resolveRepoLabel(cwd)
	repoCache.Store(cwd, label)
	return label
}

// resolveRepoLabel canonicalizes a cwd to a repo name: prefer the enclosing
// git repository (origin remote name, else git-root basename) so worktrees,
// subdirectories, and renamed checkouts of the same repo share one label.
// If cwd is not in a git repo (or no longer exists on disk), fall back to the
// old basename behavior; junk labels still route to "unknown".
func resolveRepoLabel(cwd string) string {
	if name := gitRepoName(cwd); ValidRepoName(name) {
		return name
	}
	b := filepath.Base(strings.TrimRight(cwd, "/"))
	if !ValidRepoName(b) {
		return "unknown"
	}
	return b
}

// gitRepoName walks up from dir looking for .git. On a hit it returns the
// origin remote's repo name if parseable, else the git-root directory's
// basename. Returns "" when dir is not inside a git repository.
func gitRepoName(dir string) string {
	d := filepath.Clean(dir)
	for {
		gitPath := filepath.Join(d, ".git")
		if fi, err := os.Stat(gitPath); err == nil {
			gitDir := ""
			if fi.IsDir() {
				gitDir = gitPath
			} else {
				gitDir = resolveGitdirFile(gitPath) // worktree/submodule redirection
			}
			if gitDir != "" {
				if name := originRepoName(gitDir); ValidRepoName(name) {
					return name
				}
			}
			return filepath.Base(d)
		}
		parent := filepath.Dir(d)
		if parent == d {
			return ""
		}
		d = parent
	}
}

// resolveGitdirFile follows a `gitdir: <path>` redirection file (worktrees,
// submodules) to the directory holding the repo config. Worktree gitdirs keep
// their config in the shared dir named by their `commondir` file.
func resolveGitdirFile(gitFile string) string {
	b, err := os.ReadFile(gitFile)
	if err != nil {
		return ""
	}
	target, ok := strings.CutPrefix(strings.TrimSpace(string(b)), "gitdir:")
	if !ok {
		return ""
	}
	gitDir := strings.TrimSpace(target)
	if !filepath.IsAbs(gitDir) {
		gitDir = filepath.Join(filepath.Dir(gitFile), gitDir)
	}
	gitDir = filepath.Clean(gitDir)
	if cb, err := os.ReadFile(filepath.Join(gitDir, "commondir")); err == nil {
		common := strings.TrimSpace(string(cb))
		if !filepath.IsAbs(common) {
			common = filepath.Join(gitDir, common)
		}
		gitDir = filepath.Clean(common)
	}
	return gitDir
}

// originRepoName parses <gitDir>/config for [remote "origin"]'s url and
// returns the repo name (final path segment, ".git" stripped). Handles both
// URL (https://host/owner/repo.git) and scp-like (git@host:owner/repo.git).
func originRepoName(gitDir string) string {
	b, err := os.ReadFile(filepath.Join(gitDir, "config"))
	if err != nil {
		return ""
	}
	inOrigin := false
	for _, line := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(line)
		if strings.HasPrefix(t, "[") {
			inOrigin = t == `[remote "origin"]`
			continue
		}
		if !inOrigin {
			continue
		}
		k, v, ok := strings.Cut(t, "=")
		if !ok || strings.TrimSpace(k) != "url" {
			continue
		}
		url := strings.TrimSuffix(strings.TrimSpace(v), "/")
		url = strings.TrimSuffix(url, ".git")
		// scp-like syntax: everything after the last ':' is the path part
		if i := strings.LastIndex(url, ":"); i >= 0 && !strings.Contains(url[i:], "/") {
			// e.g. host:repo (no slash after colon)
			url = url[i+1:]
		}
		if i := strings.LastIndexAny(url, "/:"); i >= 0 {
			url = url[i+1:]
		}
		return url
	}
	return ""
}

// validRepoName rejects junk repo labels — scratch dirs named after prompts or
// URLs (e.g. "2026-06-20-install-this-skill-https-github-com") make useless
// repo facets. Real repo basenames are short.
func ValidRepoName(b string) bool {
	return b != "" && b != "." && b != "/" && len(b) <= 40
}

// ---- line parsing ----

type Parsed struct {
	Text string
	Cwd  string
	Ts   float64
}

type claudeRec struct {
	Type        string          `json:"type"`
	Timestamp   string          `json:"timestamp"`
	Cwd         string          `json:"cwd"`
	IsSidechain bool            `json:"isSidechain"`
	Message     json.RawMessage `json:"message"`
}

type contentItem struct {
	Type    string          `json:"type"`
	Text    string          `json:"text"`
	Name    string          `json:"name"`
	Input   json.RawMessage `json:"input"`
	Content json.RawMessage `json:"content"`
}

func contentText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var items []contentItem
	if json.Unmarshal(raw, &items) != nil {
		return ""
	}
	var parts []string
	for _, c := range items {
		switch c.Type {
		case "text":
			if c.Text != "" {
				parts = append(parts, c.Text)
			}
		case "tool_use":
			parts = append(parts, "[tool:"+c.Name+"] "+store.Truncate(string(c.Input), maxToolChars))
		case "tool_result":
			var rs string
			if json.Unmarshal(c.Content, &rs) != nil {
				rs = string(c.Content)
			}
			parts = append(parts, "[result] "+store.Truncate(rs, maxToolChars))
		}
	}
	return strings.Join(parts, "\n")
}

func isoToEpoch(s string) float64 {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return 0
	}
	return float64(t.Unix())
}

func ParseClaudeLine(line []byte) Parsed {
	var r claudeRec
	if json.Unmarshal(line, &r) != nil {
		return Parsed{}
	}
	p := Parsed{Cwd: r.Cwd, Ts: isoToEpoch(r.Timestamp)}
	if r.Type != "user" && r.Type != "assistant" {
		return p
	}
	var msg struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(r.Message, &msg) != nil {
		return p
	}
	text := contentText(msg.Content)
	if strings.TrimSpace(text) == "" {
		return p
	}
	role := "USER"
	if r.Type == "assistant" {
		role = "ASSISTANT"
	}
	if r.IsSidechain {
		role = "SUBAGENT-" + role
	}
	p.Text = role + ": " + text
	return p
}

type codexRec struct {
	Timestamp string `json:"timestamp"`
	Type      string `json:"type"`
	Payload   struct {
		Type      string        `json:"type"`
		Cwd       string        `json:"cwd"`
		Role      string        `json:"role"`
		Name      string        `json:"name"`
		Arguments string        `json:"arguments"`
		Content   []contentItem `json:"content"`
	} `json:"payload"`
}

func ParseCodexLine(line []byte) Parsed {
	var r codexRec
	if json.Unmarshal(line, &r) != nil {
		return Parsed{}
	}
	p := Parsed{Ts: isoToEpoch(r.Timestamp)}
	switch r.Type {
	case "session_meta":
		p.Cwd = r.Payload.Cwd
	case "response_item":
		switch r.Payload.Type {
		case "message":
			var texts []string
			for _, c := range r.Payload.Content {
				if c.Text != "" {
					texts = append(texts, c.Text)
				}
			}
			body := strings.Join(texts, "\n")
			if strings.TrimSpace(body) != "" {
				role := strings.ToUpper(r.Payload.Role)
				if role == "" {
					role = "ASSISTANT"
				}
				p.Text = role + ": " + body
			}
		case "function_call":
			p.Text = "ASSISTANT: [tool:" + r.Payload.Name + "] " + store.Truncate(r.Payload.Arguments, maxToolChars)
		}
	}
	return p
}

// readNew reads from offset to EOF, rendering new complete lines into chunks.
// EndOffset only advances past complete (newline-terminated) lines.
func ReadNew(path, source string, offset int64, knownRepo string) ([]Chunk, error) {
	st, err := os.Stat(path)
	if err != nil || st.Size() <= offset {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if _, err := f.Seek(offset, 0); err != nil {
		return nil, err
	}

	parse := ParseClaudeLine
	if source == "codex" {
		parse = ParseCodexLine
	}
	repo := knownRepo
	if repo == "" {
		repo = "unknown"
	}
	session := strings.TrimSuffix(filepath.Base(path), ".jsonl")
	lastTs := float64(st.ModTime().Unix())

	var chunks []Chunk
	var pieces []string
	acc := 0
	fresh := 0 // pieces added since the last emit; the carried overlap tail is not fresh
	pos := offset

	rd := bufio.NewReaderSize(f, 1<<20)
	for {
		line, err := rd.ReadBytes('\n')
		if err != nil {
			break // EOF or partial trailing line; pick it up next cycle
		}
		pos += int64(len(line))
		p := parse(line)
		if p.Cwd != "" {
			repo = RepoFromCwd(p.Cwd)
		}
		if p.Ts > 0 {
			lastTs = p.Ts
		}
		if p.Text == "" {
			continue
		}
		pieces = append(pieces, redact(p.Text))
		fresh++
		acc += len(p.Text)
		if acc >= chunkChars {
			chunks = append(chunks, Chunk{path, source, repo, session, strings.Join(pieces, "\n\n"), pos, lastTs})
			pieces, acc = overlapTail(pieces)
			fresh = 0
		}
	}
	if fresh > 0 {
		chunks = append(chunks, Chunk{path, source, repo, session, strings.Join(pieces, "\n\n"), pos, lastTs})
	} else if len(chunks) == 0 && pos > offset {
		// only non-message lines consumed; still advance the offset
		chunks = append(chunks, Chunk{path, source, repo, session, "", pos, lastTs})
	}
	return chunks, nil
}
