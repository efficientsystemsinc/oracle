package main

// oracle CLI. Commands live in a registry (name, group, one-line summary,
// usage) that drives dispatch AND help, so `oracle help` can never drift from
// what the binary actually accepts. Every command parses through newFlagSet,
// which gives each one a uniform -h: usage line, summary, flag defaults.
// Handler bodies keep the house style: fail loud via fatal/must, print
// results plainly.

import (
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"oracle/internal/ask"
	"oracle/internal/infer"
	"oracle/internal/ingest"
	"oracle/internal/kb"
	"oracle/internal/search"
	"oracle/internal/serve"
	"oracle/internal/store"
	"oracle/internal/truth"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type command struct {
	name    string
	ns      string // "" = core, else "graph" or "admin"
	summary string // one line in help
	usage   string // full synopsis shown by -h
	run     func(db *sql.DB, args []string)
}

// commands is filled in init(): handlers reference newFlagSet → findCommand →
// commands, which as a var initializer would be an initialization cycle.
var commands []command

func init() {
	commands = []command{
		// core — the whole everyday surface; root help shows only these
		{"query", "", "hybrid lexical+semantic search over facts", "oracle query [flags] <text>", cmdQuery},
		{"ask", "", "multi-hop LLM reasoner; answers with [fact-id] citations", "oracle ask [flags] <question>", cmdAsk},
		{"install", "", "one-shot setup: daemon + local models + first ingest of your sessions", "oracle install [flags]", cmdInstall},
		{"up", "", "HTTP API on :4141 + ingest loop every 5 minutes", "oracle up [flags]", cmdUp},
		{"models", "", "show or fetch local model weights + onnxruntime", "oracle models [pull|status]", cmdModels},
		{"status", "", "db counts, last cycle, per-repo breakdown (also bare `oracle`)", "oracle status", cmdStatus},

		// graph — read the memory graph
		{"entity", "graph", "everything known about one named thing", "oracle graph entity [flags] <name>", cmdEntity},
		{"relations", "graph", "typed relations + co-mentions from an entity, n hops", "oracle graph relations [flags] <entity>", cmdGraph},
		{"brief", "graph", "standing brief: live facts for a repo, grouped by kind", "oracle graph brief [flags]", cmdBrief},
		{"topics", "graph", "k-means topic clusters over live facts", "oracle graph topics [flags]", cmdTopics},
		{"narrative", "graph", "chronological story of an entity or repo", "oracle graph narrative <entity-or-repo>", cmdNarrative},
		{"metric", "graph", "numeric time series; no name lists available metrics", "oracle graph metric [flags] [name]", cmdMetric},
		{"merge", "graph", "merge two entities; the loser's name stays as an alias", "oracle graph merge <winner> <loser>", cmdMerge},
		{"alias", "graph", "add an alias to an entity", "oracle graph alias <entity> <alias>", cmdAlias},
		{"conflicts", "graph", "open contradictions between live facts", "oracle graph conflicts", cmdConflicts},

		// admin — ingest, curation, daemon plumbing, eval & debug
		{"cycle", "admin", "one ingest pass over new session logs", "oracle admin cycle [flags]", cmdCycle},
		{"enrich", "admin", "backfill triples/metrics/entities on old facts", "oracle admin enrich [flags]", cmdEnrich},
		{"paraphrase", "admin", "backfill the paraphrase index (not run by the cycle loop)", "oracle admin paraphrase [flags]", cmdParaphrase},
		{"reembed", "admin", "(re)embed live facts into the local-model vector tables", "oracle admin reembed [flags]", cmdReembed},
		{"sweep", "admin", "re-judge similar live pairs for missed supersessions", "oracle admin sweep [flags]", cmdSweep},
		{"repair", "admin", "re-judge past supersessions; reopen the wrong ones", "oracle admin repair [flags]", cmdRepair},
		{"referee", "admin", "resolve open contradictions with the LLM judge", "oracle admin referee [flags]", cmdReferee},
		{"optimize", "admin", "merge near-duplicate entities; --relabel re-derives repo labels", "oracle admin optimize [flags]", cmdOptimize},
		{"canonpreds", "admin", "fold predicate variants into canonical forms", "oracle admin canonpreds [flags]", cmdCanonPreds},
		{"backup", "admin", "snapshot the db into ~/.oracle/backups", "oracle admin backup", cmdBackup},
		{"install-daemon", "admin", "install just the launchd/systemd service (subset of `oracle install`)", "oracle admin install-daemon", cmdInstallDaemon},
		{"init", "admin", "create the database", "oracle admin init", cmdInit},
		{"judgestats", "admin", "local-judge shadow agreement vs the LLM judge", "oracle admin judgestats", cmdJudgeStats},
		{"selfeval", "admin", "replay past supersessions as retrieval probes", "oracle admin selfeval [flags]", cmdSelfEval},
		{"askeval", "admin", "ask-confidence calibration over a probe file", "oracle admin askeval [flags]", cmdAskEval},
		{"askab", "admin", "A/B classic vs local ask on mined probes", "oracle admin askab [flags]", cmdAskAB},
		{"judgeaudit", "admin", "LLM-audit a sample of past supersession verdicts", "oracle admin judgeaudit [flags]", cmdJudgeAudit},
		{"mineprobes", "admin", "mine eval probes from the graph (TSV on stdout)", "oracle admin mineprobes [flags]", cmdMineProbes},
		{"canary", "admin", "extraction canary against fixture transcripts", "oracle admin canary [flags]", cmdCanary},
		{"fixture", "admin", "build a scrubbed fixture db for offline eval", "oracle admin fixture --out PATH", cmdFixture},
		{"chunkdump", "admin", "render session chunks for extraction-model distillation", "oracle admin chunkdump [flags]", cmdChunkDump},
	}
}

// legacyAliases maps retired top-level names to their new namespaced homes.
// They keep working forever; each use prints a one-line pointer to stderr.
var legacyAliases = map[string][2]string{
	"entity":         {"graph", "entity"},
	"graph":          {"graph", "relations"}, // `oracle graph <entity>` also still works, see nsDispatch
	"brief":          {"graph", "brief"},
	"topics":         {"graph", "topics"},
	"narrative":      {"graph", "narrative"},
	"metric":         {"graph", "metric"},
	"merge":          {"graph", "merge"},
	"alias":          {"graph", "alias"},
	"conflicts":      {"graph", "conflicts"},
	"cycle":          {"admin", "cycle"},
	"enrich":         {"admin", "enrich"},
	"paraphrase":     {"admin", "paraphrase"},
	"reembed":        {"admin", "reembed"},
	"sweep":          {"admin", "sweep"},
	"repair":         {"admin", "repair"},
	"referee":        {"admin", "referee"},
	"optimize":       {"admin", "optimize"},
	"canonpreds":     {"admin", "canonpreds"},
	"backup":         {"admin", "backup"},
	"install-daemon": {"admin", "install-daemon"},
	"init":           {"admin", "init"},
	"judgestats":     {"admin", "judgestats"},
	"selfeval":       {"admin", "selfeval"},
	"askeval":        {"admin", "askeval"},
	"askab":          {"admin", "askab"},
	"judgeaudit":     {"admin", "judgeaudit"},
	"mineprobes":     {"admin", "mineprobes"},
	"canary":         {"admin", "canary"},
	"fixture":        {"admin", "fixture"},
	"chunkdump":      {"admin", "chunkdump"},
}

func findCommand(ns, name string) *command {
	for i := range commands {
		if commands[i].ns == ns && commands[i].name == name {
			return &commands[i]
		}
	}
	return nil
}

// findByFullName resolves "query" or "graph entity" style names for help.
func findByFullName(parts []string) *command {
	if len(parts) >= 2 && (parts[0] == "graph" || parts[0] == "admin") {
		return findCommand(parts[0], parts[1])
	}
	if len(parts) >= 1 {
		return findCommand("", parts[0])
	}
	return nil
}

func main() {
	if len(os.Args) < 2 {
		// bare `oracle` = status at a glance; friendly orientation pre-setup
		if _, err := os.Stat(store.DBPath()); err != nil {
			fmt.Println("oracle — persistent memory over your AI coding sessions. no database yet at", store.DBPath())
			fmt.Println("run `oracle install` to set up the daemon, pull the local models, and ingest your recent sessions.")
			return
		}
		runCommand(findCommand("", "status"), nil)
		return
	}
	name, args := os.Args[1], os.Args[2:]

	switch name {
	case "help", "-h", "--help":
		if len(args) == 0 {
			printUsage(os.Stdout)
			return
		}
		if args[0] == "graph" || args[0] == "admin" {
			if len(args) == 1 {
				printNamespaceUsage(os.Stdout, args[0])
				return
			}
		}
		c := findByFullName(args)
		if c == nil {
			if a, ok := legacyAliases[args[0]]; ok {
				c = findCommand(a[0], a[1])
			}
		}
		if c == nil {
			unknownCommand(args[0])
		}
		// route through the command's own -h: flags are defined (and parsed)
		// before any db use, so no db is needed on this path.
		c.run(nil, []string{"-h"})
		return
	case "graph", "admin":
		nsDispatch(name, args)
		return
	}

	if c := findCommand("", name); c != nil {
		runCommand(c, args)
		return
	}
	if a, ok := legacyAliases[name]; ok {
		fmt.Fprintf(os.Stderr, "oracle: `oracle %s` is now `oracle %s %s` (old name kept as an alias)\n", name, a[0], a[1])
		runCommand(findCommand(a[0], a[1]), args)
		return
	}
	unknownCommand(name)
}

// nsDispatch handles `oracle graph <sub>` / `oracle admin <sub>`. For back
// compat, `oracle graph <entity>` (the old relations command) still works when
// <entity> is not a subcommand name.
func nsDispatch(ns string, args []string) {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		printNamespaceUsage(os.Stdout, ns)
		return
	}
	if c := findCommand(ns, args[0]); c != nil {
		runCommand(c, args[1:])
		return
	}
	if ns == "graph" {
		// old `oracle graph <entity>`: keep it working, point at the new name
		fmt.Fprintf(os.Stderr, "oracle: `oracle graph <entity>` is now `oracle graph relations <entity>` (old form kept)\n")
		runCommand(findCommand("graph", "relations"), args)
		return
	}
	fmt.Fprintf(os.Stderr, "oracle: unknown %s subcommand %q\n\n", ns, args[0])
	printNamespaceUsage(os.Stderr, ns)
	os.Exit(2)
}

func runCommand(c *command, args []string) {
	db, err := store.OpenDB()
	if err != nil {
		fatal(err)
	}
	defer db.Close()
	c.run(db, args)
}

// newFlagSet builds a command's FlagSet with a uniform -h: usage line,
// summary, then flag defaults. Every command routes its args through one so
// help behaves identically across the CLI.
func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	fs.Usage = func() {
		if c := findByShortName(name); c != nil {
			fmt.Fprintf(os.Stderr, "usage: %s\n  %s\n", c.usage, c.summary)
		} else {
			fmt.Fprintf(os.Stderr, "usage: oracle %s\n", name)
		}
		var n int
		fs.VisitAll(func(*flag.Flag) { n++ })
		if n > 0 {
			fmt.Fprintln(os.Stderr, "\nflags:")
			fs.PrintDefaults()
		}
	}
	return fs
}

// findByShortName resolves a bare command name across namespaces (names are
// unique CLI-wide); newFlagSet uses it so every handler finds its own usage.
func findByShortName(name string) *command {
	for i := range commands {
		if commands[i].name == name {
			return &commands[i]
		}
	}
	return nil
}

func printUsage(w *os.File) {
	fmt.Fprintln(w, "oracle — persistent memory over your AI coding sessions")
	fmt.Fprintln(w, "\nusage: oracle <command> [flags] [args]   (bare `oracle` = status)")
	fmt.Fprintln(w, "")
	for _, c := range commands {
		if c.ns == "" {
			fmt.Fprintf(w, "  %-8s %s\n", c.name, c.summary)
		}
	}
	fmt.Fprintln(w, "\n  graph    read the memory graph        (oracle graph — lists subcommands)")
	fmt.Fprintln(w, "  admin    ingest, curation, eval, ops   (oracle admin — lists subcommands)")
	fmt.Fprintf(w, "\nhelp: oracle help <command>  (or oracle <command> -h)\n")
	fmt.Fprintf(w, "db: %s  (override dir with ORACLE_HOME)\n", store.DBPath())
}

func printNamespaceUsage(w *os.File, ns string) {
	desc := map[string]string{
		"graph": "read the memory graph",
		"admin": "ingest, curation, daemon plumbing, eval & debug",
	}[ns]
	fmt.Fprintf(w, "oracle %s — %s\n\nusage: oracle %s <subcommand> [flags] [args]\n\n", ns, desc, ns)
	width := 0
	for _, c := range commands {
		if c.ns == ns && len(c.name) > width {
			width = len(c.name)
		}
	}
	for _, c := range commands {
		if c.ns == ns {
			fmt.Fprintf(w, "  %-*s  %s\n", width, c.name, c.summary)
		}
	}
	fmt.Fprintf(w, "\nhelp: oracle help %s <subcommand>\n", ns)
}

// unknownCommand prints the nearest name when the typo is close enough to
// call, else the full help. Exits 2 either way.
func unknownCommand(name string) {
	if s := suggest(name); s != "" {
		fmt.Fprintf(os.Stderr, "oracle: unknown command %q — did you mean %q? (oracle help lists all)\n", name, s)
	} else {
		fmt.Fprintf(os.Stderr, "oracle: unknown command %q\n\n", name)
		printUsage(os.Stderr)
	}
	os.Exit(2)
}

func suggest(name string) string {
	best, bestDist := "", 3 // suggest only within edit distance 2
	for _, c := range commands {
		if d := editDistance(name, c.name); d < bestDist {
			best, bestDist = c.name, d
		}
	}
	return best
}

func editDistance(a, b string) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, min(cur[j-1]+1, prev[j-1]+cost))
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

// --- everyday ---

func cmdInit(db *sql.DB, args []string) {
	newFlagSet("init").Parse(args)
	fmt.Println("ok:", store.DBPath())
}

func cmdStatus(db *sql.DB, args []string) {
	newFlagSet("status").Parse(args)
	var live, total, files int
	must(db.QueryRow("SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL").Scan(&live))
	must(db.QueryRow("SELECT COUNT(*) FROM facts").Scan(&total))
	must(db.QueryRow("SELECT COUNT(*) FROM files").Scan(&files))
	fmt.Printf("facts: %d live / %d total | files tracked: %d\n", live, total, files)
	var lastCycle string
	if db.QueryRow("SELECT v FROM meta WHERE k='last_cycle'").Scan(&lastCycle) == nil {
		fmt.Println("last cycle:", lastCycle)
	}
	rows, err := db.Query(`SELECT COALESCE(repo,''), COUNT(*) c FROM facts WHERE superseded_at IS NULL
		GROUP BY repo ORDER BY c DESC LIMIT 15`)
	if err == nil {
		for rows.Next() {
			var r string
			var c int
			_ = rows.Scan(&r, &c)
			fmt.Printf("  %s: %d\n", r, c)
		}
		rows.Close()
	}
}

func cmdQuery(db *sql.DB, args []string) {
	fs := newFlagSet("query")
	repo := fs.String("repo", "", "boost facts from this repo")
	k := fs.Int("k", 10, "results")
	asOf := fs.String("as-of", "", "ISO date: state of knowledge at that time")
	asJSON := fs.Bool("json", false, "json output")
	qtext := parseWithPositional(fs, args)
	if qtext == "" {
		fs.Usage()
		os.Exit(2)
	}
	asOfTs := parseAsOf(*asOf)
	hits, err := serve.QueryViaDaemonOrLocal(db, qtext, *repo, *k, asOfTs)
	if err != nil {
		fatal(err)
	}
	if *asJSON {
		b, _ := json.MarshalIndent(hits, "", " ")
		fmt.Println(string(b))
	} else {
		for _, h := range hits {
			fmt.Printf("%s[%-10s] (%s, %s, %.1fd, m=%.2f) %s\n", search.StaleMark(h), h.Kind, h.Repo, h.AsOfDate, h.AgeDays, h.Mass, h.Statement)
		}
	}
}

func cmdAsk(db *sql.DB, args []string) {
	fs := newFlagSet("ask")
	repo := fs.String("repo", "", "boost facts from this repo")
	asOf := fs.String("as-of", "", "ISO date: answer from knowledge as of then")
	qtext := parseWithPositional(fs, args)
	if qtext == "" {
		fs.Usage()
		os.Exit(2)
	}
	asOfTs := parseAsOf(*asOf)
	answer, hits, conf, err := ask.AskAuto(db, qtext, *repo, asOfTs)
	if err != nil {
		fatal(err)
	}
	fmt.Println(answer)
	fmt.Printf("\nconfidence: %.2f\n", conf.Score)
	fmt.Println("\n-- sources --")
	for _, h := range hits {
		fmt.Printf("%s[%d] (%s, %s, %s, %.1fd) %s\n", search.StaleMark(h), h.ID, h.Kind, h.Repo, h.AsOfDate, h.AgeDays, store.Truncate(h.Statement, 120))
	}
}

func cmdBrief(db *sql.DB, args []string) {
	fs := newFlagSet("brief")
	repo := fs.String("repo", "", "repo filter")
	k := fs.Int("k", 30, "facts")
	asJSON := fs.Bool("json", false, "json output")
	fs.Parse(args)
	out, err := search.Brief(db, *repo, *k)
	if err != nil {
		fatal(err)
	}
	if *asJSON {
		b, _ := json.MarshalIndent(out, "", " ")
		fmt.Println(string(b))
	} else {
		for _, kind := range []string{"preference", "decision", "gotcha", "fact", "status", "todo"} {
			if hits, ok := out[kind]; ok {
				fmt.Printf("\n## %s\n", kind)
				for _, h := range hits {
					fmt.Printf("- (%s, %.1fd) %s\n", h.Repo, h.AgeDays, h.Statement)
				}
			}
		}
	}
}

func cmdEntity(db *sql.DB, args []string) {
	fs := newFlagSet("entity")
	k := fs.Int("k", 20, "facts")
	ename := parseWithPositional(fs, args)
	if ename == "" {
		fs.Usage()
		os.Exit(2)
	}
	v, err := search.EntityView(db, ename, *k)
	if err != nil {
		fatal(err)
	}
	b, _ := json.MarshalIndent(v, "", " ")
	fmt.Println(string(b))
}

// --- graph & history ---

func cmdGraph(db *sql.DB, args []string) {
	fs := newFlagSet("graph")
	hops := fs.Int("hops", 2, "traversal depth")
	limit := fs.Int("limit", 60, "max links")
	name := parseWithPositional(fs, args)
	if name == "" {
		fs.Usage()
		os.Exit(2)
	}
	v, err := kb.Traverse(db, name, *hops, *limit)
	if err != nil {
		fatal(err)
	}
	ingest.JSONPrint(v)
}

func cmdTopics(db *sql.DB, args []string) {
	fs := newFlagSet("topics")
	k := fs.Int("k", 12, "clusters")
	fs.Parse(args)
	v, err := kb.Topics(db, *k)
	if err != nil {
		fatal(err)
	}
	ingest.JSONPrint(v)
}

func cmdNarrative(db *sql.DB, args []string) {
	fs := newFlagSet("narrative")
	name := parseWithPositional(fs, args)
	if name == "" {
		fs.Usage()
		os.Exit(2)
	}
	n, err := kb.Narrative(db, name)
	if err != nil {
		fatal(err)
	}
	fmt.Println(n)
}

func cmdMetric(db *sql.DB, args []string) {
	fs := newFlagSet("metric")
	entity := fs.String("entity", "", "filter to entity")
	name := parseWithPositional(fs, args)
	if name == "" {
		fmt.Print(ingest.MetricsList(db))
		return
	}
	v, err := ingest.MetricSeries(db, name, *entity)
	if err != nil {
		fatal(err)
	}
	ingest.JSONPrint(v)
}

func cmdConflicts(db *sql.DB, args []string) {
	newFlagSet("conflicts").Parse(args)
	must(truth.ListConflicts(db))
}

// --- ingest & curation ---

func cmdCycle(db *sql.DB, args []string) {
	fs := newFlagSet("cycle")
	maxCalls := fs.Int("max-calls", 20, "extraction LLM call budget")
	sinceDays := fs.Float64("since-days", 7, "only scan files modified in last N days")
	workers := fs.Int("workers", 4, "parallel files")
	forceRetry := fs.Bool("force-retry", false, "ignore per-file error backoff for this pass")
	fs.Parse(args)
	st, err := search.Cycle(db, *maxCalls, *sinceDays, *workers, *forceRetry)
	if err != nil {
		fatal(err)
	}
	b, _ := json.Marshal(st)
	fmt.Println(string(b))
}

func cmdEnrich(db *sql.DB, args []string) {
	fs := newFlagSet("enrich")
	maxCalls := fs.Int("max-calls", 50, "LLM batches of 40 facts")
	fs.Parse(args)
	n, err := ingest.EnrichSome(db, *maxCalls)
	if err != nil {
		fatal(fmt.Errorf("enriched %d, then: %w", n, err))
	}
	fmt.Printf("enriched %d facts\n", n)
}

func cmdParaphrase(db *sql.DB, args []string) {
	fs := newFlagSet("paraphrase")
	maxCalls := fs.Int("max-calls", 20, "LLM call budget (~20 facts per call, most-massive first)")
	fs.Parse(args)
	nFacts, calls, err := search.ParaphraseRun(db, *maxCalls)
	if err != nil {
		// partial progress is committed per batch; report it before failing
		fmt.Fprintf(os.Stderr, "paraphrase: %d facts over %d calls before error\n", nFacts, calls)
		fatal(err)
	}
	var covered, live, skipped int
	must(db.QueryRow("SELECT COUNT(*) FROM paraphrase_done d JOIN facts f ON f.id = d.fact_id WHERE f.superseded_at IS NULL").Scan(&covered))
	must(db.QueryRow("SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL").Scan(&live))
	must(db.QueryRow("SELECT COUNT(*) FROM paraphrase_skip").Scan(&skipped))
	fmt.Printf("paraphrased %d facts in %d calls | coverage: %d/%d live facts | %d skipped (shape-failed)\n", nFacts, calls, covered, live, skipped)
}

func cmdReembed(db *sql.DB, args []string) {
	fs := newFlagSet("reembed")
	batch := fs.Int("batch", 64, "texts per embed call")
	fs.Parse(args)
	must(search.Reembed(db, *batch))
}

func cmdSweep(db *sql.DB, args []string) {
	fs := newFlagSet("sweep")
	workers := fs.Int("workers", 8, "parallel judge calls")
	perEntity := fs.Int("per-entity", 6, "max candidate pairs per entity (highest similarity kept)")
	maxPairs := fs.Int("max-pairs", 2000, "global candidate pair cap")
	dryRun := fs.Bool("dry-run", false, "print verdicts, write nothing")
	fs.Parse(args)
	st, err := truth.Sweep(db, *workers, *perEntity, *maxPairs, *dryRun)
	must(err)
	b, _ := json.Marshal(st)
	fmt.Println(string(b))
}

func cmdRepair(db *sql.DB, args []string) {
	fs := newFlagSet("repair")
	workers := fs.Int("workers", 8, "parallel LLM workers (pairs partitioned by old_id % N)")
	sample := fs.Int("sample", 0, "audit-only dry run: judge first N pairs, print verdicts, write nothing")
	maxPairs := fs.Int("max-pairs", 0, "stop after N pairs (0 = all)")
	fs.Parse(args)
	must(truth.RunRepair(db, *workers, *sample, *maxPairs))
}

func cmdReferee(db *sql.DB, args []string) {
	fs := newFlagSet("referee")
	workers := fs.Int("workers", 4, "parallel LLM judgments")
	dryRun := fs.Bool("dry-run", false, "judge only; apply nothing, record nothing")
	fs.Parse(args)
	counts, err := truth.Referee(db, *workers, *dryRun)
	for _, k := range []string{"SUPERSEDE_NEWER", "DIFFERENT_SCOPE", "UNRESOLVED", "ERROR"} {
		if counts[k] > 0 {
			fmt.Printf("%s: %d\n", k, counts[k])
		}
	}
	must(err)
}

func cmdMerge(db *sql.DB, args []string) {
	fs := newFlagSet("merge")
	fs.Parse(args)
	rest := fs.Args()
	if len(rest) < 2 {
		fs.Usage()
		os.Exit(2)
	}
	must(kb.MergeEntities(db, rest[0], rest[1]))
}

func cmdAlias(db *sql.DB, args []string) {
	fs := newFlagSet("alias")
	fs.Parse(args)
	rest := fs.Args()
	if len(rest) < 2 {
		fs.Usage()
		os.Exit(2)
	}
	id, disp, ok := kb.ResolveEntity(db, rest[0])
	if !ok {
		fatal(fmt.Errorf("entity %q not found", rest[0]))
	}
	must(kb.AddAlias(db, id, rest[1], 1.0))
	fmt.Printf("alias %q -> %s\n", rest[1], disp)
}

func cmdOptimize(db *sql.DB, args []string) {
	fs := newFlagSet("optimize")
	apply := fs.Bool("apply", false, "apply variant merges / relabels (default: suggest only)")
	relabel := fs.Bool("relabel", false, "re-derive repo labels from session cwds via git (dry-run unless --apply)")
	fs.Parse(args)
	if *relabel {
		// relabel-only pass: the standard optimize deletes rows even
		// without --apply, so a relabel dry-run must not run it
		out, err := search.RelabelRepos(db, *apply)
		fmt.Print(out)
		must(err)
		return
	}
	out, err := search.Optimize(db, *apply)
	fmt.Print(out)
	must(err)
}

func cmdCanonPreds(db *sql.DB, args []string) {
	fs := newFlagSet("canonpreds")
	apply := fs.Bool("apply", false, "apply the fold (default: dry-run print)")
	fs.Parse(args)
	must(truth.CanonPreds(db, *apply))
}

func cmdBackup(db *sql.DB, args []string) {
	newFlagSet("backup").Parse(args)
	dest, err := store.BackupDB(db)
	must(err)
	fmt.Println("backup:", dest)
}

// --- local models ---

func cmdModels(db *sql.DB, args []string) {
	fs := newFlagSet("models")
	force := fs.Bool("force", false, "re-download weights even when already present")
	fs.Parse(args)
	sub := "status"
	if rest := fs.Args(); len(rest) > 0 {
		sub = rest[0]
	}
	switch sub {
	case "pull":
		must(modelsPull(*force))
		infer.ModelsStatus()
	case "status":
		infer.ModelsStatus()
	default:
		fatal(fmt.Errorf("unknown models subcommand %q (want pull or status)", sub))
	}
}

// modelsPull downloads any missing platform model assets (all of them when
// force is true) and ensures onnxruntime is present.
func modelsPull(force bool) error {
	m, err := infer.LoadManifest()
	if err != nil {
		return err
	}
	dirs := infer.PlatformAssets(m)
	if !force {
		var missing []string
		for _, d := range dirs {
			if _, err := os.Stat(filepath.Join(infer.ModelsDir(), d)); err != nil {
				missing = append(missing, d)
			}
		}
		dirs = missing
	}
	if len(dirs) > 0 {
		if err := infer.PullModels(m, dirs); err != nil {
			return err
		}
	}
	return infer.EnsureORT()
}

func cmdJudgeStats(db *sql.DB, args []string) {
	newFlagSet("judgestats").Parse(args)
	must(truth.RunJudgeStats(db))
}

// --- daemon ---

func cmdUp(db *sql.DB, args []string) {
	fs := newFlagSet("up")
	port := fs.Int("port", 4141, "listen port")
	maxCalls := fs.Int("max-calls", 20, "extraction call budget per cycle")
	noLoop := fs.Bool("no-loop", false, "serve only, no ingest loop")
	fs.Parse(args)
	fatal(serve.ServeHTTP(db, *port, *maxCalls, !*noLoop))
}

func cmdInstallDaemon(db *sql.DB, args []string) {
	newFlagSet("install-daemon").Parse(args)
	installDaemon()
}

// cmdInstall is the whole first-run setup in one command: install the
// background daemon, pull the local models, ingest the user's most recent
// agent sessions, build the local vector index, then prove the memory works
// by querying something oracle just learned about them.
func cmdInstall(db *sql.DB, args []string) {
	fs := newFlagSet("install")
	sinceDays := fs.Float64("since-days", 2, "how far back the first ingest looks")
	maxCalls := fs.Int("max-calls", 10, "extraction LLM call budget for the first ingest")
	fs.Parse(args)

	fmt.Println("[1/4] installing the background daemon (`oracle up`)...")
	installDaemon()

	fmt.Println("[2/4] pulling local model weights + onnxruntime...")
	must(modelsPull(false))

	fmt.Println("[3/4] reading your recent agent sessions...")
	if n := len(ingest.Discover(*sinceDays)); n == 0 {
		fmt.Printf("no agent-session history found in the last %.0f days.\n", *sinceDays)
		fmt.Printf("oracle reads Claude Code logs from %s and codex logs from %s automatically —\n", ingest.ClaudeDir(), ingest.CodexDir())
		fmt.Println("run a few agent sessions, then `oracle admin cycle` (the daemon also ingests every 5 minutes).")
	} else if st, err := search.Cycle(db, *maxCalls, *sinceDays, 4, false); err != nil {
		fmt.Fprintf(os.Stderr, "first ingest failed: %v\n", err)
		fmt.Fprintln(os.Stderr, "if this is a key problem: set ORACLE_AZURE_KEY (or put the key in ~/.oracle/azure.key), then run `oracle admin cycle`.")
	} else {
		fmt.Printf("ingested %d new facts from %d session files\n", st.Facts, st.FilesSeen)
	}

	fmt.Println("[4/4] building the local vector index...")
	var live, vecs int
	must(db.QueryRow("SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL").Scan(&live))
	must(db.QueryRow("SELECT COUNT(*) FROM fact_vecs_local").Scan(&vecs))
	if live > 0 && vecs == 0 {
		os.Setenv("ORACLE_LOCAL_EMBED", "1") // local corpus must come from the local model
		if err := search.Reembed(db, 64); err != nil {
			fmt.Fprintf(os.Stderr, "reembed failed (rerun later with `oracle admin reembed`): %v\n", err)
		}
	} else {
		fmt.Println("vector index already populated — skipping")
	}

	installWow(db)

	fmt.Println("\nnext:")
	fmt.Println("  oracle                          # status at a glance")
	fmt.Println("  oracle query \"<anything>\"       # search everything your agents have done")
	fmt.Println("  oracle ask \"what did I decide about <topic>?\"   # cited answers over your history")
}

// installWow proves the install worked with the user's own data: take the
// top entity of the highest-mass live fact and show what oracle knows.
func installWow(db *sql.DB) {
	var ent string
	err := db.QueryRow(`SELECT e.display FROM facts f
		JOIN fact_entities fe ON fe.fact_id = f.id
		JOIN entities e ON e.id = fe.entity_id
		WHERE f.superseded_at IS NULL
		ORDER BY f.mass DESC, e.seen_count DESC LIMIT 1`).Scan(&ent)
	if err != nil {
		return // nothing ingested yet; the [3/4] step already explained why
	}
	hits, err := serve.QueryViaDaemonOrLocal(db, ent, "", 5, 0)
	if err != nil || len(hits) == 0 {
		return
	}
	fmt.Printf("\noracle already remembers this from your sessions (query: %q):\n", ent)
	for _, h := range hits {
		fmt.Printf("%s[%-10s] (%s, %.1fd) %s\n", search.StaleMark(h), h.Kind, h.Repo, h.AgeDays, store.Truncate(h.Statement, 140))
	}
}

// --- eval & debug ---

func cmdSelfEval(db *sql.DB, args []string) {
	fs := newFlagSet("selfeval")
	sample := fs.Int("sample", 25, "supersession pairs to replay")
	k := fs.Int("k", 10, "hit@k / recall@k cutoff")
	asJSON := fs.Bool("json", false, "json output")
	verbose := fs.Bool("v", false, "print each miss (pair ids + old statement)")
	fixture := fs.String("fixture", "", "eval against this fixture db instead of the live one")
	fs.Parse(args)
	evalDB := db
	if *fixture != "" {
		if _, err := os.Stat(*fixture); err != nil {
			fatal(fmt.Errorf("fixture: %w", err))
		}
		fdb, err := store.OpenFixtureDB(*fixture)
		if err != nil {
			fatal(err)
		}
		defer fdb.Close()
		evalDB = fdb
	}
	ev, err := search.RunSelfEval(evalDB, *sample, *k, *verbose)
	if err != nil {
		fatal(err)
	}
	if *asJSON {
		ingest.JSONPrint(ev)
	} else {
		search.PrintSelfEval(ev)
	}
}

func cmdAskEval(db *sql.DB, args []string) {
	fs := newFlagSet("askeval")
	probes := fs.String("probes", "eval/ask_confidence_probes.json", "probes file")
	out := fs.String("out", "", "write per-question JSONL here")
	thr := fs.Float64("threshold", ask.AskAbstainThreshold, "abstain threshold to evaluate")
	split := fs.String("split", "", "restrict to build|holdout (default: all)")
	fs.Parse(args)
	must(ask.RunAskEval(db, *probes, *out, *thr, *split))
}

func cmdAskAB(db *sql.DB, args []string) {
	fs := newFlagSet("askab")
	probes := fs.String("probes", "eval/probes_1k.tsv", "mined probe TSV")
	n := fs.Int("n", 40, "probes to sample")
	fs.Parse(args)
	must(ask.RunAskAB(db, *probes, *n))
}

func cmdJudgeAudit(db *sql.DB, args []string) {
	fs := newFlagSet("judgeaudit")
	sample := fs.Int("sample", 50, "number of supersession pairs to audit")
	workers := fs.Int("workers", 8, "parallel judge calls")
	fs.Parse(args)
	_, err := truth.JudgeAudit(db, *sample, *workers)
	must(err)
}

func cmdMineProbes(db *sql.DB, args []string) {
	fs := newFlagSet("mineprobes")
	n := fs.Int("n", 150, "probes to mine (TSV on stdout; redirect to eval/probes_mined.tsv)")
	fs.Parse(args)
	must(search.MineProbes(db, *n))
}

func cmdCanary(db *sql.DB, args []string) {
	fs := newFlagSet("canary")
	dir := fs.String("dir", "testdata/canary_chunks", "directory of transcript fixtures")
	expected := fs.String("expected", "testdata/canary_expected.json", "expected keyword sets")
	gen := fs.Bool("gen", false, "print extracted statements instead of judging (for authoring expected)")
	fs.Parse(args)
	must(ingest.RunCanary(*dir, *expected, *gen))
}

func cmdFixture(db *sql.DB, args []string) {
	fs := newFlagSet("fixture")
	out := fs.String("out", "", "output path for the fixture db (required)")
	fs.Parse(args)
	if *out == "" {
		fs.Usage()
		os.Exit(2)
	}
	must(store.BuildFixture(store.DBPath(), *out))
}

func cmdChunkDump(db *sql.DB, args []string) {
	chunkDump(args)
}

// --- shared helpers ---

// parseAsOf converts an --as-of ISO date to a unix timestamp (0 = unset).
func parseAsOf(s string) float64 {
	if s == "" {
		return 0
	}
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		fatal(fmt.Errorf("--as-of: %w (want YYYY-MM-DD)", err))
	}
	return float64(t.Unix())
}

// parseWithPositional parses flags around one positional argument, so both
// `oracle query --as-of 2026-06-20 "text"` and `oracle query "text" -k 5`
// work. Two passes: the first stops at the positional, the second consumes
// any flags after it. (The old hand-rolled splitter grabbed the first
// non-dash token, which stole flag VALUES — `--as-of 2026-06-20 "text"`
// queried for "2026-06-20".) Extra positionals are a loud error.
func parseWithPositional(fs *flag.FlagSet, args []string) string {
	fs.Parse(args)
	rest := fs.Args()
	if len(rest) == 0 {
		return ""
	}
	pos := rest[0]
	fs.Parse(rest[1:])
	if extra := fs.Args(); len(extra) > 0 {
		fatal(fmt.Errorf("unexpected argument %q after %q (quote multi-word text)", extra[0], pos))
	}
	return pos
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "oracle:", err)
	os.Exit(1)
}

func must(err error) {
	if err != nil {
		fatal(err)
	}
}

func installDaemon() {
	exe, err := os.Executable()
	if err != nil {
		fatal(err)
	}
	if runtime.GOOS == "linux" {
		installDaemonLinux(exe)
		return
	}
	installDaemonDarwin(exe)
}

// launchdPlist renders the macOS LaunchAgent for `oracle up`.
func launchdPlist(exe, logPath string) string {
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.oracle.daemon</string>
  <key>ProgramArguments</key><array><string>%s</string><string>up</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
</dict></plist>`, exe, logPath, logPath)
}

func installDaemonDarwin(exe string) {
	home, _ := os.UserHomeDir()
	plist := filepath.Join(home, "Library", "LaunchAgents", "com.oracle.daemon.plist")
	if err := os.WriteFile(plist, []byte(launchdPlist(exe, filepath.Join(store.OracleHome(), "daemon.log"))), 0o644); err != nil {
		fatal(err)
	}
	_ = exec.Command("launchctl", "unload", plist).Run()
	if err := exec.Command("launchctl", "load", plist).Run(); err != nil {
		fatal(err)
	}
	fmt.Println("loaded", plist, "-> http://127.0.0.1:4141")
}

// systemdUnit renders the user-level unit for `oracle up`.
func systemdUnit(exe, logPath string) string {
	return fmt.Sprintf(`[Unit]
Description=oracle bi-temporal fact graph daemon

[Service]
ExecStart=%s up
Restart=always
RestartSec=5
StandardOutput=append:%s
StandardError=append:%s

[Install]
WantedBy=default.target
`, exe, logPath, logPath)
}

func installDaemonLinux(exe string) {
	home, _ := os.UserHomeDir()
	unitDir := filepath.Join(home, ".config", "systemd", "user")
	if err := os.MkdirAll(unitDir, 0o755); err != nil {
		fatal(err)
	}
	unit := filepath.Join(unitDir, "oracle.service")
	if err := os.WriteFile(unit, []byte(systemdUnit(exe, filepath.Join(store.OracleHome(), "daemon.log"))), 0o644); err != nil {
		fatal(err)
	}
	for _, args := range [][]string{{"daemon-reload"}, {"enable", "--now", "oracle.service"}} {
		cmd := exec.Command("systemctl", append([]string{"--user"}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			fatal(fmt.Errorf("systemctl --user %s: %v: %s", strings.Join(args, " "), err, out))
		}
	}
	fmt.Println("loaded", unit, "-> http://127.0.0.1:4141")
	fmt.Println("note: run `loginctl enable-linger` once so the daemon survives logout")
}
