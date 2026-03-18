# Legacy Tier 3 RAGAS Archive

This folder freezes the retired Tier 3 synthetic testset-generation path.

Archived here:
- the old RAGAS generator module
- its smoke runner
- its raw checkpoint artifact
- its generator-specific tests

These files are preserved for reference only.

Do not wire the active eval workflow, docs, or CI to this archive.
The supported Tier 3 benchmark path is the curated manual-seed benchmark in
`eval/golden/v1/tier3_rag.jsonl` plus the manual authoring helpers.
