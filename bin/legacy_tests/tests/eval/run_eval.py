from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import app


def lexical_overlap(a: str, b: str) -> float:
    set_a = {t for t in a.lower().split() if len(t) > 2}
    set_b = {t for t in b.lower().split() if len(t) > 2}
    if not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_b)


def run_eval(dataset_path: str = 'tests/eval/testset_medical.jsonl') -> dict:
    client = TestClient(app)

    rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    relevance_scores = []
    faithfulness_scores = []

    for row in rows:
        response = client.post('/api/chat', json={'message': row['question']})
        payload = response.json()
        answer = payload.get('response', '')

        relevance_scores.append(lexical_overlap(answer, row['question']))

        # Simple groundedness proxy: citations or trusted source mention.
        has_grounding = ('[Source:' in answer) or ('source' in payload)
        faithfulness_scores.append(1.0 if has_grounding else 0.0)

    result = {
        'num_samples': len(rows),
        'answer_relevance': sum(relevance_scores) / max(len(relevance_scores), 1),
        'faithfulness_proxy': sum(faithfulness_scores) / max(len(faithfulness_scores), 1),
    }

    out = Path('tests/eval/last_eval_result.json')
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(run_eval(), indent=2))
