package ingest

// Tests for watch.go.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

func writeSessionFile(t *testing.T, msgs []string) string {
	t.Helper()
	var b strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&b, `{"type":"user","timestamp":"2026-07-01T10:00:00Z","cwd":"/home/sam/quasar","message":{"content":%q}}`+"\n", m)
	}
	path := filepath.Join(t.TempDir(), "session.jsonl")
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadNewChunkOverlap(t *testing.T) {
	var msgs []string
	for i := 0; i < 40; i++ {
		msgs = append(msgs, fmt.Sprintf("MSG-%04d %s", i, strings.Repeat("x", 1200)))
	}
	path := writeSessionFile(t, msgs)
	chunks, err := ReadNew(path, "claude", 0, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}

	markerRe := regexp.MustCompile(`MSG-\d{4}`)
	for i := 1; i < len(chunks); i++ {
		prev := markerRe.FindAllString(chunks[i-1].Text, -1)
		curr := markerRe.FindAllString(chunks[i].Text, -1)
		if len(prev) == 0 || len(curr) == 0 {
			t.Fatalf("chunk %d or %d has no markers", i-1, i)
		}
		// the cut is softened: chunk i starts with the tail of chunk i-1
		if curr[0] != prev[len(prev)-1] {
			t.Errorf("chunk %d should start with chunk %d's last message; got %s vs %s",
				i, i-1, curr[0], prev[len(prev)-1])
		}
		// but must contain fresh content beyond the overlap
		prevSet := map[string]bool{}
		for _, m := range prev {
			prevSet[m] = true
		}
		freshCount := 0
		for _, m := range curr {
			if !prevSet[m] {
				freshCount++
			}
		}
		if freshCount == 0 {
			t.Errorf("chunk %d is pure overlap", i)
		}
		if chunks[i].EndOffset <= chunks[i-1].EndOffset {
			t.Errorf("offsets must strictly increase: %d then %d", chunks[i-1].EndOffset, chunks[i].EndOffset)
		}
	}

	st, _ := os.Stat(path)
	last := chunks[len(chunks)-1]
	if last.EndOffset != st.Size() {
		t.Errorf("final EndOffset %d != file size %d", last.EndOffset, st.Size())
	}
	if last.Repo != "quasar" {
		t.Errorf("repo not derived from cwd: %q", last.Repo)
	}

	// resuming from a committed offset must not replay consumed lines
	resumed, err := ReadNew(path, "claude", chunks[0].EndOffset, "quasar")
	if err != nil {
		t.Fatal(err)
	}
	all := markerRe.FindAllString(chunks[0].Text, -1)
	firstAfterResume := markerRe.FindAllString(resumed[0].Text, -1)[0]
	if firstAfterResume != fmt.Sprintf("MSG-%04d", len(all)) {
		t.Errorf("resume from offset should start at MSG-%04d, got %s", len(all), firstAfterResume)
	}
}

func TestParseClaudeLine(t *testing.T) {
	p := ParseClaudeLine([]byte(`{"type":"user","timestamp":"2026-07-01T10:00:00Z","cwd":"/home/sam/quasar","message":{"content":"hello there"}}`))
	if p.Text != "USER: hello there" {
		t.Errorf("user string content: got %q", p.Text)
	}
	wantTs := float64(time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC).Unix())
	if p.Cwd != "/home/sam/quasar" || p.Ts != wantTs {
		t.Errorf("cwd/ts: got %q %v, want ts %v", p.Cwd, p.Ts, wantTs)
	}

	p = ParseClaudeLine([]byte(`{"type":"assistant","message":{"content":[{"type":"text","text":"running it"},{"type":"tool_use","name":"Bash","input":{"command":"ls"}},{"type":"tool_result","content":"file-a\nfile-b"}]}}`))
	for _, want := range []string{"ASSISTANT: running it", "[tool:Bash]", `"command"`, "[result] file-a"} {
		if !strings.Contains(p.Text, want) {
			t.Errorf("assistant content array: missing %q in %q", want, p.Text)
		}
	}

	p = ParseClaudeLine([]byte(`{"type":"user","isSidechain":true,"message":{"content":"sub task"}}`))
	if !strings.HasPrefix(p.Text, "SUBAGENT-USER: ") {
		t.Errorf("sidechain prefix: got %q", p.Text)
	}

	// non-message records still carry cwd/ts but no text
	p = ParseClaudeLine([]byte(`{"type":"summary","cwd":"/home/sam/oracle","timestamp":"2026-07-01T10:00:00Z"}`))
	if p.Text != "" || p.Cwd != "/home/sam/oracle" {
		t.Errorf("summary record: got text %q cwd %q", p.Text, p.Cwd)
	}

	if p = ParseClaudeLine([]byte(`not json at all`)); p.Text != "" || p.Cwd != "" || p.Ts != 0 {
		t.Errorf("garbage line must parse to zero value, got %+v", p)
	}
}

func TestParseCodexLine(t *testing.T) {
	p := ParseCodexLine([]byte(`{"timestamp":"2026-07-01T10:00:00Z","type":"session_meta","payload":{"cwd":"/home/sam/quasar"}}`))
	if p.Cwd != "/home/sam/quasar" || p.Text != "" {
		t.Errorf("session_meta: got %+v", p)
	}

	p = ParseCodexLine([]byte(`{"type":"response_item","payload":{"type":"message","role":"user","content":[{"text":"fix the eval"}]}}`))
	if p.Text != "USER: fix the eval" {
		t.Errorf("codex message: got %q", p.Text)
	}

	p = ParseCodexLine([]byte(`{"type":"response_item","payload":{"type":"message","content":[{"text":"done"}]}}`))
	if p.Text != "ASSISTANT: done" {
		t.Errorf("missing role should default to ASSISTANT: got %q", p.Text)
	}

	p = ParseCodexLine([]byte(`{"type":"response_item","payload":{"type":"function_call","name":"shell","arguments":"{\"cmd\":\"ls\"}"}}`))
	if !strings.Contains(p.Text, "[tool:shell]") {
		t.Errorf("function_call: got %q", p.Text)
	}
}

func TestRedact(t *testing.T) {
	secrets := []string{
		"key is sk-abcdefghij0123456789",
		"aws AKIAABCDEFGHIJKLMNOP",
		"slack xoxb-1234567890-abcdef",
		"github ghp_abcdefghij0123456789",
		"jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdefghijklm.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwxyz012345",
		"password=hunter2hunter2",
		"api_key: 0123456789abcdef",
		"MY_SECRET = supersecretvalue",
	}
	for _, s := range secrets {
		if got := redact(s); !strings.Contains(got, "[REDACTED]") {
			t.Errorf("not redacted: %q -> %q", s, got)
		}
	}
	benign := "set statement_cache_size=0 for asyncpg behind pgbouncer on box atlas01"
	if got := redact(benign); got != benign {
		t.Errorf("benign text mangled: %q", got)
	}
}

func TestRepoFromCwd(t *testing.T) {
	// nonexistent cwds fall back to basename behavior; junk still -> unknown
	cases := map[string]string{
		"":                                     "unknown",
		"/":                                    "unknown",
		"/nonexistent-oracle-test/sam/quasar":  "quasar",
		"/nonexistent-oracle-test/sam/quasar/": "quasar",
		// prompt-derived scratch dirs are junk labels, not repos
		"/nonexistent-oracle-test/az/2026-06-20-install-this-skill-https-github-com": "unknown",
	}
	for in, want := range cases {
		if got := RepoFromCwd(in); got != want {
			t.Errorf("repoFromCwd(%q) = %q, want %q", in, got, want)
		}
	}
}

// mkGitRepo creates <root>/<name>/.git with a config carrying an origin url.
func mkGitRepo(t *testing.T, root, name, originURL string) string {
	t.Helper()
	gitDir := filepath.Join(root, name, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := "[core]\n\trepositoryformatversion = 0\n"
	if originURL != "" {
		cfg += "[remote \"origin\"]\n\turl = " + originURL + "\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
	}
	if err := os.WriteFile(filepath.Join(gitDir, "config"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(root, name)
}

func TestRepoFromCwdGit(t *testing.T) {
	root := t.TempDir()

	// origin remote name wins over the checkout dir's basename
	repo := mkGitRepo(t, root, "checkout-v1", "https://github.com/sam/quasar.git")
	if got := RepoFromCwd(repo); got != "quasar" {
		t.Errorf("https origin: got %q, want quasar", got)
	}
	// subdirectory of the repo resolves to the same label
	sub := filepath.Join(repo, "deep", "nested", "dir")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := RepoFromCwd(sub); got != "quasar" {
		t.Errorf("subdir: got %q, want quasar", got)
	}

	// scp-like origin url
	scp := mkGitRepo(t, root, "scpclone", "git@github.com:sam/oracle.git")
	if got := RepoFromCwd(scp); got != "oracle" {
		t.Errorf("scp origin: got %q, want oracle", got)
	}

	// no origin remote -> git-root basename, even from a subdir
	noRemote := mkGitRepo(t, root, "localonly", "")
	nsub := filepath.Join(noRemote, "pkg")
	if err := os.MkdirAll(nsub, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := RepoFromCwd(nsub); got != "localonly" {
		t.Errorf("no origin: got %q, want localonly", got)
	}

	// worktree: .git FILE redirecting to main/.git/worktrees/<n> with commondir
	main := mkGitRepo(t, root, "mainco", "https://github.com/sam/quasar.git")
	wtGitDir := filepath.Join(main, ".git", "worktrees", "wt1")
	if err := os.MkdirAll(wtGitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wtGitDir, "commondir"), []byte("../..\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wt := filepath.Join(root, "wt1-checkout")
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wt, ".git"), []byte("gitdir: "+wtGitDir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := RepoFromCwd(wt); got != "quasar" {
		t.Errorf("worktree: got %q, want quasar", got)
	}

	// cache: second lookup of an already-resolved cwd returns the same label
	if got := RepoFromCwd(repo); got != "quasar" {
		t.Errorf("cached lookup: got %q, want quasar", got)
	}
}

func TestOverlapTail(t *testing.T) {
	if got, total := overlapTail([]string{strings.Repeat("a", overlapChars+1)}); len(got) != 0 || total != 0 {
		t.Errorf("oversized trailing piece should give no overlap, got %d pieces total %d", len(got), total)
	}
	pieces := []string{strings.Repeat("a", 1500), strings.Repeat("b", 800), strings.Repeat("c", 900)}
	got, total := overlapTail(pieces)
	if total != 1700 || total > overlapChars {
		t.Errorf("want total 1700 within cap, got %d", total)
	}
	if len(got) != 2 || got[0][0] != 'b' || got[1][0] != 'c' {
		t.Errorf("expected trailing [b,c] pieces, got %d pieces", len(got))
	}
}
