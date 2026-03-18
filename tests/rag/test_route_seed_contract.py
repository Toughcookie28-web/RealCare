import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / 'data' / 'route_seeds_raw.json'
EXPECTED_ROUTES = {
    'vector',
    'chitchat',
    'web',
    'memory',
    'literature',
    'future_clinical_db',
}


def test_route_seeds_raw_has_expected_route_coverage_and_counts():
    data = json.loads(RAW_PATH.read_text())

    assert set(data) == EXPECTED_ROUTES
    for route, seeds in data.items():
        assert len(seeds) == 10, route
        assert all(isinstance(seed, str) and seed.strip() for seed in seeds), route
        assert len(seeds) == len(set(seeds)), route
