"""
Schema validator and QA report generator for golden evaluation datasets.

Validates all JSONL tier files against schema_v1, checks quality gates,
and produces golden_v1_manifest.json + qa_report.json.

Usage:
    python -m eval.validate              # validate + report
    python -m eval.validate --strict     # exit 1 on any gate failure
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / 'golden' / 'v1'

TIER_FILES = {
    'guardrail': 'tier1_guardrail.jsonl',
    'routing': 'tier2_routing.jsonl',
    'rag': 'tier3_rag.jsonl',
    'conversational': 'tier4_conversation.jsonl',
    'robustness': 'tier5_robustness.jsonl',
}

VALID_TIERS = set(TIER_FILES.keys())
VALID_BEHAVIORS = {
    'block', 'answer', 'answer_with_citations',
    'educational_pivot', 'resolve_coreference_then_answer', 'chitchat',
}
VALID_GUARDRAIL_ACTIONS = {'safety_refusal', 'crisis_resources', None}
VALID_DIFFICULTIES = {'simple', 'reasoning', 'multi_context', 'adversarial'}
VALID_SPLITS = {'train', 'dev', 'test'}
VALID_ROUTES = {'chitchat', 'vector', 'web', 'literature', 'future_clinical_db', None}
VALID_SOURCE_TYPES = {'human_seed', 'llm_generated', 'ragas_generated', 'mined_pattern'}
VALID_REVIEW_STATUSES = {'draft', 'approved', 'rejected'}
REQUIRED_TIER1_CATEGORIES = {
    'harmful_synthesis',
    'self_harm_crisis',
    'prompt_injection',
    'diagnosis_seeking',
    'legitimate_sensitive',
    'jailbreak_roleplay',
    'indirect_harm',
    'medical_misinformation',
    'dosage_boundary',
    'child_emergency',
    'professional_impersonation',
}
REQUIRED_TIER4_CATEGORIES = {
    'coreference_resolution',
    'fact_recall',
    'summary_continuity',
    'topic_switching',
    'correction_handling',
    'progressive_disclosure',
    'contradictory_statements',
    'clarification_loop',
}

# Quality gate thresholds
MIN_SAMPLES_PER_TIER1_CATEGORY = 4
MIN_CHAINS_PER_TIER4_CATEGORY = 2
BLOCK_RATIO_MIN = 0.40
BLOCK_RATIO_MAX = 0.60
MAX_NEAR_DUPLICATE_RATE = 0.05
REQUIRED_EVIDENCE_KEYS = {"required", "supporting", "hard_negative_candidates"}
REQUIRED_EVIDENCE_ANCHOR_KEYS = {
    "chunk_id",
    "doc_id",
    "page",
    "section",
    "content_type",
    "anchor_text",
}


def load_tier(tier: str) -> list[dict]:
    path = GOLDEN_DIR / TIER_FILES[tier]
    if not path.exists():
        return []
    samples = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            samples.append({'_parse_error': str(exc), '_line': i, '_file': str(path)})
    return samples


def validate_sample(sample: dict, expected_tier: str) -> list[str]:
    errors: list[str] = []
    sid = sample.get('id', '???')

    if '_parse_error' in sample:
        return [f"{sid}: JSON parse error on line {sample['_line']}: {sample['_parse_error']}"]

    # Required fields
    for field in (
        'id',
        'version',
        'tier',
        'category',
        'question',
        'expected_planned_route',
        'expected_final_route',
        'expected_behavior',
        'difficulty',
        'split',
        'tags',
        'provenance',
    ):
        if field not in sample:
            errors.append(f"{sid}: missing required field '{field}'")

    if sample.get('version') != 'v1':
        errors.append(f"{sid}: version must be 'v1', got '{sample.get('version')}'")

    if sample.get('tier') != expected_tier:
        errors.append(f"{sid}: tier mismatch, expected '{expected_tier}', got '{sample.get('tier')}'")

    if expected_tier == 'guardrail' and sample.get('category') not in REQUIRED_TIER1_CATEGORIES:
        errors.append(f"{sid}: invalid guardrail category '{sample.get('category')}'")
    if expected_tier == 'conversational' and sample.get('category') not in REQUIRED_TIER4_CATEGORIES:
        errors.append(f"{sid}: invalid conversational category '{sample.get('category')}'")

    if sample.get('expected_behavior') not in VALID_BEHAVIORS:
        errors.append(f"{sid}: invalid expected_behavior '{sample.get('expected_behavior')}'")

    if sample.get('difficulty') not in VALID_DIFFICULTIES:
        errors.append(f"{sid}: invalid difficulty '{sample.get('difficulty')}'")

    if sample.get('split') not in VALID_SPLITS:
        errors.append(f"{sid}: invalid split '{sample.get('split')}'")

    # Route validation
    for route_field in ('expected_planned_route', 'expected_final_route'):
        val = sample.get(route_field)
        if val not in VALID_ROUTES:
            errors.append(f"{sid}: invalid {route_field} '{val}'")

    # Guardrail action consistency
    behavior = sample.get('expected_behavior')
    action = sample.get('expected_guardrail_action')
    if behavior == 'block' and action not in ('safety_refusal', 'crisis_resources'):
        errors.append(f"{sid}: block behavior requires expected_guardrail_action")
    if behavior != 'block' and action is not None:
        errors.append(f"{sid}: non-block behavior should have null expected_guardrail_action")

    # Conversational tier requires multi_turn_context
    if expected_tier == 'conversational':
        ctx = sample.get('multi_turn_context')
        if not isinstance(ctx, list) or len(ctx) == 0:
            errors.append(f"{sid}: conversational tier requires non-empty multi_turn_context list")

    # RAG tier requires ground_truth_contexts
    if expected_tier == 'rag':
        contexts = sample.get('ground_truth_contexts')
        if not isinstance(contexts, list) or len(contexts) == 0:
            errors.append(f"{sid}: rag tier requires non-empty ground_truth_contexts")
        evidence = sample.get('evidence')
        if not isinstance(evidence, dict):
            errors.append(f"{sid}: rag tier requires evidence object")
        else:
            if set(evidence.keys()) != REQUIRED_EVIDENCE_KEYS:
                errors.append(f"{sid}: rag evidence keys must equal {sorted(REQUIRED_EVIDENCE_KEYS)}")
            required = evidence.get('required')
            supporting = evidence.get('supporting')
            hard_negatives = evidence.get('hard_negative_candidates')
            if not isinstance(required, list) or not required:
                errors.append(f"{sid}: rag evidence.required must be a non-empty list")
                required = []
            if not isinstance(supporting, list):
                errors.append(f"{sid}: rag evidence.supporting must be a list")
                supporting = []
            if not isinstance(hard_negatives, list):
                errors.append(f"{sid}: rag evidence.hard_negative_candidates must be a list")
                hard_negatives = []

            gold_chunk_ids = set(sample.get('ground_truth_chunk_ids') or [])
            evidence_gold_ids: set[str] = set()
            for label, anchors in (
                ('required', required),
                ('supporting', supporting),
                ('hard_negative_candidates', hard_negatives),
            ):
                for idx, anchor in enumerate(anchors):
                    if not isinstance(anchor, dict):
                        errors.append(f"{sid}: rag evidence.{label}[{idx}] must be an object")
                        continue
                    if set(anchor.keys()) != REQUIRED_EVIDENCE_ANCHOR_KEYS:
                        errors.append(
                            f"{sid}: rag evidence.{label}[{idx}] keys must equal "
                            f"{sorted(REQUIRED_EVIDENCE_ANCHOR_KEYS)}"
                        )
                        continue
                    if not anchor.get('anchor_text'):
                        errors.append(f"{sid}: rag evidence.{label}[{idx}].anchor_text must be non-empty")
                    chunk_id = anchor.get('chunk_id')
                    if label in {'required', 'supporting'}:
                        evidence_gold_ids.add(chunk_id)
                        if gold_chunk_ids and chunk_id not in gold_chunk_ids:
                            errors.append(
                                f"{sid}: rag evidence.{label}[{idx}].chunk_id '{chunk_id}' "
                                "must be present in ground_truth_chunk_ids"
                            )
                    elif gold_chunk_ids and chunk_id in gold_chunk_ids:
                        errors.append(
                            f"{sid}: rag evidence.hard_negative_candidates[{idx}].chunk_id '{chunk_id}' "
                            "must not overlap ground_truth_chunk_ids"
                        )

            if gold_chunk_ids and not evidence_gold_ids.issubset(gold_chunk_ids):
                errors.append(f"{sid}: rag evidence anchors must be a subset of ground_truth_chunk_ids")

    # Provenance validation
    prov = sample.get('provenance', {})
    if not isinstance(prov, dict):
        errors.append(f"{sid}: provenance must be an object")
    else:
        if prov.get('source_type') not in VALID_SOURCE_TYPES:
            errors.append(f"{sid}: invalid provenance.source_type '{prov.get('source_type')}'")
        if prov.get('review_status') not in VALID_REVIEW_STATUSES:
            errors.append(f"{sid}: invalid provenance.review_status '{prov.get('review_status')}'")

    # Tags must be a list
    if not isinstance(sample.get('tags'), list):
        errors.append(f"{sid}: tags must be a list")

    return errors


def check_near_duplicates(all_samples: list[dict]) -> list[tuple[str, str, float]]:
    """Simple lexical near-duplicate check using Jaccard similarity on question tokens."""
    duplicates = []
    questions = [(s.get('id', '?'), set(s.get('question', '').lower().split()))
                 for s in all_samples if '_parse_error' not in s]

    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            id_a, tokens_a = questions[i]
            id_b, tokens_b = questions[j]
            if not tokens_a or not tokens_b:
                continue
            intersection = len(tokens_a & tokens_b)
            union = len(tokens_a | tokens_b)
            jaccard = intersection / union if union else 0.0
            if jaccard >= 0.92:
                duplicates.append((id_a, id_b, round(jaccard, 4)))

    return duplicates


def compute_file_hash(path: Path) -> str:
    if not path.exists():
        return 'missing'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_validation(strict: bool = False) -> dict:
    all_errors: list[str] = []
    all_samples: list[dict] = []
    tier_stats: dict[str, dict] = {}
    gate_results: dict[str, bool] = {}

    # Load and validate each tier
    for tier, filename in TIER_FILES.items():
        samples = load_tier(tier)
        tier_errors = []
        for sample in samples:
            tier_errors.extend(validate_sample(sample, tier))
        all_errors.extend(tier_errors)
        all_samples.extend(samples)

        categories = Counter(s.get('category', 'unknown') for s in samples if '_parse_error' not in s)
        tier_stats[tier] = {
            'file': filename,
            'count': len(samples),
            'errors': len(tier_errors),
            'categories': dict(categories),
            'file_exists': (GOLDEN_DIR / filename).exists(),
        }

    # Gate 1: 100% schema compliance
    gate_results['schema_compliance'] = len(all_errors) == 0

    # Gate 2: All tiers non-empty
    gate_results['all_tiers_populated'] = all(
        tier_stats[t]['file_exists'] and tier_stats[t]['count'] > 0 for t in TIER_FILES
    )

    # Gate 3: Tier 1 block/non-block ratio
    t1_samples = [s for s in all_samples if s.get('tier') == 'guardrail' and '_parse_error' not in s]
    t1_block = sum(1 for s in t1_samples if s.get('expected_behavior') == 'block')
    t1_total = len(t1_samples)
    if t1_total > 0:
        block_ratio = t1_block / t1_total
        gate_results['tier1_block_ratio'] = BLOCK_RATIO_MIN <= block_ratio <= BLOCK_RATIO_MAX
    else:
        gate_results['tier1_block_ratio'] = False

    # Gate 4: Near-duplicate rate
    duplicates = check_near_duplicates(all_samples)
    valid_count = sum(1 for s in all_samples if '_parse_error' not in s)
    dup_rate = len(duplicates) / max(valid_count, 1)
    gate_results['near_duplicate_rate'] = dup_rate <= MAX_NEAR_DUPLICATE_RATE

    # Gate 5: Review completion for generated samples
    generated = [s for s in all_samples
                 if '_parse_error' not in s
                 and s.get('provenance', {}).get('source_type') in ('llm_generated', 'ragas_generated')]
    unreviewed = [s for s in generated if s.get('provenance', {}).get('review_status') == 'draft']
    gate_results['review_completion'] = len(unreviewed) == 0

    # Gate 6: Ambiguous routing tagged
    t2_samples = [s for s in all_samples if s.get('tier') == 'routing' and '_parse_error' not in s]
    ambiguous = [s for s in t2_samples if 'ambiguous' in s.get('tags', [])]
    gate_results['ambiguous_routing_tagged'] = len(ambiguous) > 0 if len(t2_samples) > 0 else True

    # Gate 7: Manifest generated (checked at end)
    gate_results['manifest_generated'] = True  # will be true after this runs

    # Gate 8a: Tier 1 category minimums
    t1_categories = Counter(s.get('category') for s in t1_samples)
    t1_under_min = {
        cat: t1_categories.get(cat, 0)
        for cat in sorted(REQUIRED_TIER1_CATEGORIES)
        if t1_categories.get(cat, 0) < MIN_SAMPLES_PER_TIER1_CATEGORY
    }
    gate_results['tier1_category_coverage'] = len(t1_under_min) == 0

    # Gate 8b: Tier 4 category minimums
    t4_samples = [s for s in all_samples if s.get('tier') == 'conversational' and '_parse_error' not in s]
    t4_categories = Counter(s.get('category') for s in t4_samples)
    t4_under_min = {
        cat: t4_categories.get(cat, 0)
        for cat in sorted(REQUIRED_TIER4_CATEGORIES)
        if t4_categories.get(cat, 0) < MIN_CHAINS_PER_TIER4_CATEGORY
    }
    gate_results['tier4_category_coverage'] = len(t4_under_min) == 0

    all_gates_pass = all(gate_results.values())

    # Build QA report
    qa_report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'golden_dir': str(GOLDEN_DIR),
        'total_samples': len(all_samples),
        'valid_samples': valid_count,
        'total_errors': len(all_errors),
        'errors': all_errors[:100],  # cap for readability
        'tier_stats': tier_stats,
        'quality_gates': gate_results,
        'all_gates_pass': all_gates_pass,
        'near_duplicates': [
            {'id_a': a, 'id_b': b, 'jaccard': j} for a, b, j in duplicates
        ],
        'tier1_block_ratio': {
            'block': t1_block,
            'non_block': t1_total - t1_block,
            'total': t1_total,
            'ratio': round(t1_block / max(t1_total, 1), 4),
        },
        'tier1_category_gaps': t1_under_min,
        'tier4_category_gaps': t4_under_min,
        'unreviewed_generated_samples': [s.get('id') for s in unreviewed],
    }

    # Write QA report
    qa_path = GOLDEN_DIR / 'qa_report.json'
    qa_path.write_text(json.dumps(qa_report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Build and write manifest
    manifest = {
        'version': 'v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_samples': valid_count,
        'tier_counts': {t: tier_stats[t]['count'] for t in TIER_FILES},
        'file_hashes': {
            filename: compute_file_hash(GOLDEN_DIR / filename)
            for filename in TIER_FILES.values()
        },
        'schema_version': 'schema_v1',
        'all_gates_pass': all_gates_pass,
    }
    manifest_path = GOLDEN_DIR / 'golden_v1_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    # Console output
    print(f"\n{'='*60}")
    print(f"  Golden Dataset QA Report")
    print(f"{'='*60}")
    print(f"  Total samples:  {len(all_samples)}")
    print(f"  Valid samples:  {valid_count}")
    print(f"  Schema errors:  {len(all_errors)}")
    print()
    for tier, stats in tier_stats.items():
        status = 'OK' if stats['file_exists'] else 'MISSING'
        print(f"  {tier:<15} {stats['count']:>4} samples  [{status}]")
    print()
    print(f"  Quality Gates:")
    for gate, passed in gate_results.items():
        icon = 'PASS' if passed else 'FAIL'
        print(f"    [{icon}] {gate}")
    print()

    if all_errors:
        print(f"  First {min(len(all_errors), 10)} errors:")
        for err in all_errors[:10]:
            print(f"    - {err}")
        print()

    if duplicates:
        print(f"  Near-duplicates ({len(duplicates)}):")
        for a, b, j in duplicates[:5]:
            print(f"    {a} <-> {b}  (jaccard={j})")
        print()

    if t1_under_min:
        print(f"  Tier 1 categories below minimum ({MIN_SAMPLES_PER_TIER1_CATEGORY}):")
        for cat, cnt in t1_under_min.items():
            print(f"    {cat}: {cnt}")
        print()

    result_line = 'ALL GATES PASSED' if all_gates_pass else 'SOME GATES FAILED'
    print(f"  Result: {result_line}")
    print(f"{'='*60}\n")

    return qa_report


def main():
    strict = '--strict' in sys.argv
    report = run_validation(strict=strict)
    if strict and not report['all_gates_pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
