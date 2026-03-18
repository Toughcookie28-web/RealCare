import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_knn_route_returns_majority_vote():
    from tools.knn_router import knn_route

    seeds = [
        ([1.0, 0.01, 0.0], "vector"),
        ([0.99, 0.02, 0.0], "vector"),
        ([0.98, 0.03, 0.0], "vector"),
        ([0.01, 1.0, 0.0], "chitchat"),
        ([0.02, 0.99, 0.0], "chitchat"),
    ]
    result = knn_route([1.0, 0.0, 0.0], seeds, k=3)
    assert result == "vector"

    result = knn_route([0.0, 1.0, 0.0], seeds, k=3)
    assert result == "chitchat"


def test_knn_route_returns_default_on_empty_seeds():
    from tools.knn_router import knn_route

    result = knn_route([1.0, 0.0], [], k=3)
    assert result == "vector"


def test_load_seeds_returns_empty_on_missing_file():
    from tools.knn_router import load_seeds

    seeds = load_seeds("/nonexistent/path/seeds.json")
    assert seeds == []


def test_load_seeds_parses_valid_file():
    from tools.knn_router import load_seeds

    data = {
        "vector": [{"query": "what is diabetes", "embedding": [0.1, 0.2]}],
        "chitchat": [{"query": "hello", "embedding": [0.3, 0.4]}],
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        f.flush()
        seeds = load_seeds(f.name)

    assert len(seeds) == 2
    assert seeds[0] == ([0.1, 0.2], "vector")
    assert seeds[1] == ([0.3, 0.4], "chitchat")


def test_knn_fallback_route_returns_vector_without_seeds():
    from tools.knn_router import knn_fallback_route

    state = {'optimized_query': 'test', 'question': 'test'}
    result = knn_fallback_route(state)
    assert result == 'vector'
