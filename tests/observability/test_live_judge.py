import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import observability.live_judge as live_judge


class _FakeCounter:
    def __init__(self):
        self.count = 0

    def labels(self, **kwargs):
        return self

    def inc(self):
        self.count += 1


class _FakeHistogram:
    def __init__(self):
        self.values = []

    def labels(self, **kwargs):
        return self

    def observe(self, value):
        self.values.append(value)


class _FakeDocument:
    def __init__(self, page_content):
        self.page_content = page_content


def test_build_live_judge_payload_extracts_contexts():
    payload = live_judge.build_live_judge_payload(
        question="What are the side effects of metformin?",
        trace_id="trace-1",
        result={
            "generation": "Common effects include nausea and diarrhea.",
            "route": "vector",
            "documents": [
                _FakeDocument("Metformin commonly causes gastrointestinal upset."),
                _FakeDocument("It is widely used for type 2 diabetes."),
            ],
        },
    )

    assert payload["question"] == "What are the side effects of metformin?"
    assert payload["answer"] == "Common effects include nausea and diarrhea."
    assert payload["route"] == "vector"
    assert payload["trace_id"] == "trace-1"
    assert payload["contexts"] == [
        "Metformin commonly causes gastrointestinal upset.",
        "It is widely used for type 2 diabetes.",
    ]


def test_live_judge_returns_none_when_disabled():
    service = live_judge.LiveJudgeService(enabled=False, sampling_ratio=1.0)

    assert service.judge(
        {
            "question": "What is metformin?",
            "answer": "It is a diabetes drug.",
            "contexts": ["Metformin is used in type 2 diabetes."],
            "route": "vector",
            "trace_id": "trace-2",
        }
    ) is None


def test_live_judge_records_metrics_on_success(monkeypatch):
    monkeypatch.setattr(live_judge, "LIVE_JUDGED_REQUESTS", _FakeCounter())
    monkeypatch.setattr(live_judge, "LIVE_JUDGE_FAILURES", _FakeCounter())
    monkeypatch.setattr(live_judge, "LIVE_JUDGE_LATENCY", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_ANSWER_RELEVANCE", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_GROUNDEDNESS", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_CONTEXT_PRECISION_PROXY", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_CONTEXT_COVERAGE_PROXY", _FakeHistogram())

    service = live_judge.LiveJudgeService(
        enabled=True,
        sampling_ratio=1.0,
        scorer=lambda payload: {
            "answer_relevance": 0.8,
            "groundedness": 0.9,
            "context_precision_proxy": 0.7,
            "context_coverage_proxy": 0.6,
        },
    )

    result = service.judge(
        {
            "question": "What is metformin?",
            "answer": "It is used for diabetes.",
            "contexts": ["Metformin is used for type 2 diabetes."],
            "route": "vector",
            "trace_id": "trace-3",
        }
    )

    assert result["answer_relevance"] == 0.8
    assert live_judge.LIVE_JUDGED_REQUESTS.count == 1
    assert live_judge.LIVE_JUDGE_FAILURES.count == 0
    assert len(live_judge.LIVE_JUDGE_LATENCY.values) == 1
    assert live_judge.LIVE_ANSWER_RELEVANCE.values == [0.8]
    assert live_judge.LIVE_GROUNDEDNESS.values == [0.9]
    assert live_judge.LIVE_CONTEXT_PRECISION_PROXY.values == [0.7]
    assert live_judge.LIVE_CONTEXT_COVERAGE_PROXY.values == [0.6]


def test_live_judge_records_failure_metric_on_error(monkeypatch):
    monkeypatch.setattr(live_judge, "LIVE_JUDGED_REQUESTS", _FakeCounter())
    monkeypatch.setattr(live_judge, "LIVE_JUDGE_FAILURES", _FakeCounter())
    monkeypatch.setattr(live_judge, "LIVE_JUDGE_LATENCY", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_ANSWER_RELEVANCE", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_GROUNDEDNESS", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_CONTEXT_PRECISION_PROXY", _FakeHistogram())
    monkeypatch.setattr(live_judge, "LIVE_CONTEXT_COVERAGE_PROXY", _FakeHistogram())

    def _raise(_payload):
        raise RuntimeError("judge failed")

    service = live_judge.LiveJudgeService(
        enabled=True,
        sampling_ratio=1.0,
        scorer=_raise,
    )

    result = service.judge(
        {
            "question": "What is metformin?",
            "answer": "It is used for diabetes.",
            "contexts": ["Metformin is used for type 2 diabetes."],
            "route": "vector",
            "trace_id": "trace-4",
        }
    )

    assert result is None
    assert live_judge.LIVE_JUDGED_REQUESTS.count == 0
    assert live_judge.LIVE_JUDGE_FAILURES.count == 1
