import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings import Settings


def test_working_memory_and_live_judge_settings_contract():
    settings = Settings(_env_file=None, debug=False)

    assert isinstance(settings.working_memory_recent_turns_ttl, int)
    assert settings.working_memory_recent_turns_ttl > 0

    assert isinstance(settings.working_memory_core_state_ttl, int)
    assert settings.working_memory_core_state_ttl > 0

    assert isinstance(settings.working_memory_recent_turns_max, int)
    assert settings.working_memory_recent_turns_max > 0

    assert isinstance(settings.live_judge_enabled, bool)

    assert isinstance(settings.live_judge_sampling_ratio, float)
    assert 0.0 <= settings.live_judge_sampling_ratio <= 1.0


def test_long_term_memory_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings(_env_file=None)
    assert s.long_term_memory_enabled is False
    assert s.long_term_memory_recall_k == 3
    assert s.long_term_memory_min_confidence == 0.6
    assert s.long_term_memory_episode_min_facts == 1


def test_diversity_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings(_env_file=None)
    assert s.diversity_mmr_enabled is True
    assert s.diversity_mmr_lambda == 0.7
    assert s.diversity_max_per_section == 2
