"""oracle CLI: init | cycle | query | brief | status | up | install-daemon"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(prog="oracle", description="bi-temporal memory over all agent sessions")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    c = sub.add_parser("cycle")
    c.add_argument("--max-calls", type=int, default=20)
    c.add_argument("--since-days", type=float, default=7)
    q = sub.add_parser("query")
    q.add_argument("q")
    q.add_argument("--repo")
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--as-of", help="ISO date: state of knowledge at that time")
    q.add_argument("--json", action="store_true")
    b = sub.add_parser("brief")
    b.add_argument("--repo")
    b.add_argument("-k", type=int, default=30)
    b.add_argument("--json", action="store_true")
    u = sub.add_parser("up")
    u.add_argument("--port", type=int, default=4141)
    u.add_argument("--max-calls", type=int, default=20)
    sub.add_parser("install-daemon")
    a = p.parse_args()

    from . import db, graph
    if a.cmd == "init":
        db.connect().close()
        print(f"ok: {db.DB_PATH}")
    elif a.cmd == "status":
        con = db.connect()
        live = con.execute("SELECT COUNT(*) c FROM facts WHERE superseded_at IS NULL").fetchone()["c"]
        tot = con.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
        nf = con.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
        last = con.execute("SELECT v FROM meta WHERE k='last_cycle'").fetchone()
        by_repo = con.execute(
            "SELECT repo, COUNT(*) c FROM facts WHERE superseded_at IS NULL "
            "GROUP BY repo ORDER BY c DESC LIMIT 15").fetchall()
        print(f"facts: {live} live / {tot} total | files tracked: {nf}")
        if last:
            print(f"last cycle: {last['v']}")
        for r in by_repo:
            print(f"  {r['repo']}: {r['c']}")
    elif a.cmd == "cycle":
        print(json.dumps(graph.cycle(max_calls=a.max_calls, since_days=a.since_days)))
    elif a.cmd == "query":
        as_of = None
        if a.as_of:
            from datetime import datetime
            as_of = datetime.fromisoformat(a.as_of).timestamp()
        con = db.connect()
        hits = graph.search(con, a.q, repo=a.repo, k=a.k, as_of=as_of)
        if a.json:
            print(json.dumps(hits, indent=1))
        else:
            for h in hits:
                print(f"[{h['kind']:^10}] ({h['repo']}, {h['age_days']}d, m={h['mass']}) {h['statement']}")
    elif a.cmd == "brief":
        con = db.connect()
        out = graph.brief(con, a.repo, a.k)
        if a.json:
            print(json.dumps(out, indent=1))
        else:
            for kind in ("preference", "decision", "gotcha", "fact", "status", "todo"):
                if kind in out:
                    print(f"\n## {kind}")
                    for h in out[kind]:
                        print(f"- ({h['repo']}, {h['age_days']}d) {h['statement']}")
    elif a.cmd == "up":
        from . import serve
        serve.up(port=a.port, max_calls=a.max_calls)
    elif a.cmd == "install-daemon":
        _install_daemon()


def _install_daemon():
    import os
    import subprocess
    plist = os.path.expanduser("~/Library/LaunchAgents/com.sam.oracle.plist")
    exe = subprocess.run([sys.executable, "-c", "import shutil;print(shutil.which('oracle'))"],
                         capture_output=True, text=True).stdout.strip()
    if not exe or exe == "None":
        raise SystemExit("oracle not on PATH; pip install -e first")
    log = os.path.expanduser("~/.oracle/daemon.log")
    with open(plist, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.sam.oracle</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>up</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict></plist>""")
    subprocess.run(["launchctl", "unload", plist], capture_output=True)
    subprocess.run(["launchctl", "load", plist], check=True)
    print(f"loaded {plist} -> http://127.0.0.1:4141")


if __name__ == "__main__":
    main()
