from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is importable when executed as a file path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_contracts import validate_rag_runtime_contract


def main() -> int:
    errors = validate_rag_runtime_contract()
    if errors:
        print("RAG runtime preflight failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("RAG runtime preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
