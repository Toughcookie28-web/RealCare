#!/usr/bin/env python3
"""Embed hand-written seed queries and save to data/route_seeds.json.

Usage:
    python scripts/generate_route_seeds.py

Reads seed queries from data/route_seeds_raw.json (format: {"route": ["query1", ...]}).
Embeds each query using embed_query() and writes to data/route_seeds.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / 'data' / 'route_seeds_raw.json'
OUT_PATH = ROOT / 'data' / 'route_seeds.json'


def main():
    from tools.embedding_client import embed_query

    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found. Create it with seed queries per route.")
        print('Format: {"vector": ["query1", ...], "chitchat": ["query2", ...]}')
        sys.exit(1)

    with open(RAW_PATH) as f:
        raw = json.load(f)

    output: dict[str, list[dict]] = {}
    total = 0
    for route, queries in raw.items():
        output[route] = []
        for query in queries:
            embedding = embed_query(query)
            output[route].append({"query": query, "embedding": embedding})
            total += 1
        print(f"  {route}: {len(queries)} seeds embedded")

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f)

    print(f"Saved {total} embedded seeds to {OUT_PATH}")


if __name__ == '__main__':
    main()
