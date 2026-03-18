import sys
from pathlib import Path

# Ensure repository root is importable when executed as a file path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import init_database


if __name__ == '__main__':
    init_database()
    print('Database initialized.')
