#!/usr/bin/env python3
"""Render the live oracle entity co-mention graph to assets/graph.png.

Reads ~/.oracle/oracle.db (read-only), takes the top-N entities by
seen_count, keeps co-mention edges between them, lays the graph out with
graphviz sfdp, and draws a minimal monochrome picture: node size by degree,
labels only on the most-connected nodes. Entity names only — no fact text.

Usage: python3 scripts/render_graph.py [--db PATH] [--out assets/graph.png]
"""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import subprocess
import tempfile

TOP_ENTITIES = 150
LABELED = 40
WIDTH_PX = 2000


def load(db: str):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ents = con.execute(
        "SELECT id, display, seen_count FROM entities "
        "ORDER BY seen_count DESC LIMIT ?",
        (TOP_ENTITIES,),
    ).fetchall()
    ids = {e[0] for e in ents}
    edges = [
        (a, b, c)
        for a, b, c in con.execute("SELECT a, b, count FROM entity_edges")
        if a in ids and b in ids and a != b
    ]
    con.close()
    return ents, edges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.path.expanduser("~/.oracle/oracle.db"))
    p.add_argument("--out", default="assets/graph.png")
    args = p.parse_args()

    ents, edges = load(args.db)
    deg: dict[int, int] = {}
    for a, b, _ in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1

    # keep only nodes with at least one edge
    ents = [e for e in ents if deg.get(e[0])]
    labeled = {
        e[0] for e in sorted(ents, key=lambda e: -deg[e[0]])[:LABELED]
    }

    lines = [
        "graph oracle {",
        '  graph [bgcolor="white", outputorder="edgesfirst", overlap="prism",'
        '         overlap_scaling=-14, splines="line", pad=0.5, K=1.6, repulsiveforce=2.0];',
        '  node  [shape="circle", style="filled", color="none",'
        '         fontname="Helvetica", fixedsize=true];',
        '  edge  [color="#00000018"];',
    ]
    max_deg = max(deg.values())
    for eid, name, _seen in ents:
        d = deg[eid]
        r = 0.05 + 0.28 * math.sqrt(d / max_deg)
        if eid in labeled:
            fs = 10 + 12 * math.sqrt(d / max_deg)
            label = name.replace('"', "")
            lines.append(
                f'  n{eid} [width={r:.3f}, fillcolor="#1a1a1a", label="",'
                f' xlabel="{label}", fontsize={fs:.1f}, fontcolor="#1a1a1a"];'
            )
        else:
            lines.append(
                f'  n{eid} [width={r:.3f}, fillcolor="#b5b5b5", label=""];'
            )
    maxc = max(c for _, _, c in edges)
    for a, b, c in edges:
        w = 0.2 + 1.2 * (c / maxc) ** 0.5
        lines.append(f"  n{a} -- n{b} [penwidth={w:.2f}];")
    lines.append("}")

    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write("\n".join(lines))
        dot = f.name
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    subprocess.run(
        ["sfdp", f"-Gdpi={WIDTH_PX // 20}", "-Gsize=20,20!", "-Tpng",
         dot, "-o", args.out],
        check=True,
    )
    os.unlink(dot)
    print(f"{args.out}: {len(ents)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()
