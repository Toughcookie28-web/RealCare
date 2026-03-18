from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.embedding_bootstrap import (
    DOCLING_ARTIFACTS_DIRNAME,
    DOCLING_LAYOUT_MODEL_FILENAME,
    ensure_docling_artifacts,
)


def test_ensure_docling_artifacts_downloads_missing_layout_model(tmp_path):
    calls: list[Path] = []

    def fake_download_models(*, output_dir, **_kwargs):
        output_path = Path(output_dir)
        calls.append(output_path)
        (output_path / DOCLING_LAYOUT_MODEL_FILENAME).write_text("ok", encoding="utf-8")

    artifacts_dir = ensure_docling_artifacts(str(tmp_path), download_models_fn=fake_download_models)

    assert artifacts_dir == tmp_path / DOCLING_ARTIFACTS_DIRNAME
    assert calls == [artifacts_dir]
    assert (artifacts_dir / DOCLING_LAYOUT_MODEL_FILENAME).exists()


def test_ensure_docling_artifacts_skips_download_when_layout_model_exists(tmp_path):
    artifacts_dir = tmp_path / DOCLING_ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / DOCLING_LAYOUT_MODEL_FILENAME).write_text("ok", encoding="utf-8")
    called = False

    def fake_download_models(**_kwargs):
        nonlocal called
        called = True

    resolved = ensure_docling_artifacts(str(tmp_path), download_models_fn=fake_download_models)

    assert resolved == artifacts_dir
    assert called is False


def test_ensure_docling_artifacts_surfaces_nested_layout_snapshot(tmp_path):
    artifacts_dir = tmp_path / DOCLING_ARTIFACTS_DIRNAME

    def fake_download_models(*, output_dir, **_kwargs):
        nested = Path(output_dir) / "docling-project--docling-layout-heron"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / DOCLING_LAYOUT_MODEL_FILENAME).write_text("ok", encoding="utf-8")
        (nested / "config.json").write_text("{}", encoding="utf-8")
        (nested / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    resolved = ensure_docling_artifacts(str(tmp_path), download_models_fn=fake_download_models)

    assert resolved == artifacts_dir
    assert (artifacts_dir / DOCLING_LAYOUT_MODEL_FILENAME).exists()
    assert (artifacts_dir / "config.json").exists()
    assert (artifacts_dir / "preprocessor_config.json").exists()
