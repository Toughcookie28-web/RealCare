from __future__ import annotations

import inspect
import os
from pathlib import Path
import shutil
import sys

DEFAULT_EMBEDDING_CACHE_DIR = "/app/.cache/embeddings"
DOCLING_ARTIFACTS_DIRNAME = "docling_artifacts"
RAPIDOCR_MODELS_DIRNAME = "rapidocr/models"
DOCLING_LAYOUT_MODEL_FILENAME = "model.safetensors"


def resolve_embedding_cache_root(cache_dir: str | None) -> Path:
    return Path((cache_dir or DEFAULT_EMBEDDING_CACHE_DIR).strip() or DEFAULT_EMBEDDING_CACHE_DIR)


def embedding_cache_path(cache_dir: str | None, relative_path: str = "") -> Path:
    root = resolve_embedding_cache_root(cache_dir)
    return root / relative_path if relative_path else root


def configure_embedding_cache_env(cache_dir: str | None) -> str:
    root = resolve_embedding_cache_root(cache_dir)
    rapidocr_cache = embedding_cache_path(cache_dir, "rapidocr")
    rapidocr_models = embedding_cache_path(cache_dir, RAPIDOCR_MODELS_DIRNAME)
    docling_artifacts = embedding_cache_path(cache_dir, DOCLING_ARTIFACTS_DIRNAME)
    hf_home = embedding_cache_path(cache_dir, "hf_home")
    hf_cache = embedding_cache_path(cache_dir, "huggingface_hub")
    tf_cache = embedding_cache_path(cache_dir, "transformers")
    hf_modules = embedding_cache_path(cache_dir, "hf_modules")
    hf_assets = embedding_cache_path(cache_dir, "hf_assets")
    st_cache = embedding_cache_path(cache_dir, "sentence_transformers")
    xdg_cache = embedding_cache_path(cache_dir, "xdg")
    xdg_data = embedding_cache_path(cache_dir, "xdg_data")
    xdg_config = embedding_cache_path(cache_dir, "xdg_config")
    torch_home = embedding_cache_path(cache_dir, "torch")
    tmp_home = embedding_cache_path(cache_dir, "home")

    for path in (
        root,
        rapidocr_cache,
        rapidocr_models,
        docling_artifacts,
        hf_home,
        hf_cache,
        tf_cache,
        hf_modules,
        hf_assets,
        st_cache,
        xdg_cache,
        xdg_data,
        xdg_config,
        torch_home,
        tmp_home,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["EMBEDDING_CACHE_DIR"] = str(root)
    os.environ["DOCLING_ARTIFACTS_DIR"] = str(docling_artifacts)
    os.environ["RAPIDOCR_MODELS_DIR"] = str(rapidocr_models)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(tf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_modules)
    os.environ["HF_ASSETS_CACHE"] = str(hf_assets)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_cache)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    os.environ["XDG_DATA_HOME"] = str(xdg_data)
    os.environ["XDG_CONFIG_HOME"] = str(xdg_config)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["HOME"] = str(tmp_home)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    _patch_imported_cache_constants()
    return str(st_cache)


def ensure_docling_artifacts(cache_dir: str | None, download_models_fn=None) -> Path:
    artifacts_dir = embedding_cache_path(cache_dir, DOCLING_ARTIFACTS_DIRNAME)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    layout_model = artifacts_dir / DOCLING_LAYOUT_MODEL_FILENAME
    if layout_model.exists():
        return artifacts_dir

    download_models_fn = download_models_fn or _resolve_docling_downloader()
    _download_docling_models(download_models_fn, artifacts_dir)
    _surface_nested_docling_layout_snapshot(artifacts_dir)

    if not layout_model.exists():
        raise FileNotFoundError(f"Missing safe tensors file: {layout_model}")

    return artifacts_dir


def _resolve_docling_downloader():
    try:
        from docling.utils.model_downloader import download_models  # type: ignore

        return download_models
    except Exception:
        from docling.utils import model_downloader  # type: ignore

        return model_downloader.download_models


def _download_docling_models(download_models_fn, artifacts_dir: Path) -> None:
    options = {
        "output_dir": artifacts_dir,
        "with_layout": True,
        "with_tableformer": True,
        "with_easyocr": False,
        "with_code_formula": False,
        "with_picture_classifier": False,
    }
    try:
        params = inspect.signature(download_models_fn).parameters
    except Exception:
        download_models_fn(**options)
        return

    kwargs = {
        name: value
        for name, value in options.items()
        if name in params
    }
    if "output_dir" not in kwargs and "artifacts_path" in params:
        kwargs["artifacts_path"] = artifacts_dir
    if "output_dir" not in kwargs and "artifacts_path" not in kwargs:
        kwargs["output_dir"] = artifacts_dir

    download_models_fn(**kwargs)


def _surface_nested_docling_layout_snapshot(artifacts_dir: Path) -> None:
    root_layout_model = artifacts_dir / DOCLING_LAYOUT_MODEL_FILENAME
    if root_layout_model.exists():
        return

    nested_candidates = sorted(
        artifacts_dir.glob(f"**/{DOCLING_LAYOUT_MODEL_FILENAME}"),
        key=lambda path: len(path.parts),
    )
    nested_layout_model = next((path for path in nested_candidates if path.parent != artifacts_dir), None)
    if nested_layout_model is None:
        return

    for child in nested_layout_model.parent.iterdir():
        if not child.is_file():
            continue
        target = artifacts_dir / child.name
        if target.exists():
            continue
        _link_or_copy(child, target)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _patch_imported_cache_constants() -> None:
    try:
        hub_constants = sys.modules.get("huggingface_hub.constants")
        if hub_constants is not None:
            setattr(hub_constants, "HF_HOME", os.environ["HF_HOME"])
            setattr(hub_constants, "HF_HUB_CACHE", os.environ["HF_HUB_CACHE"])
            setattr(hub_constants, "HUGGINGFACE_HUB_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
            setattr(hub_constants, "HF_ASSETS_CACHE", os.environ["HF_ASSETS_CACHE"])
    except Exception:
        pass

    try:
        tf_hub = sys.modules.get("transformers.utils.hub")
        if tf_hub is not None:
            setattr(tf_hub, "TRANSFORMERS_CACHE", os.environ["TRANSFORMERS_CACHE"])
            setattr(tf_hub, "HF_MODULES_CACHE", os.environ["HF_MODULES_CACHE"])
    except Exception:
        pass

    try:
        tf_dynamic = sys.modules.get("transformers.dynamic_module_utils")
        if tf_dynamic is not None:
            setattr(tf_dynamic, "HF_MODULES_CACHE", os.environ["HF_MODULES_CACHE"])
    except Exception:
        pass
