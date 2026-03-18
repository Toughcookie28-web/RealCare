from pathlib import Path
import sys
import types
import importlib.util


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pdf_parser_module():
    settings_module = types.ModuleType("core.settings")
    settings_module.get_settings = lambda: types.SimpleNamespace(embedding_cache_dir="/tmp/medigenius-test")

    bootstrap_module = types.ModuleType("tools.embedding_bootstrap")
    bootstrap_module.configure_embedding_cache_env = lambda cache_dir: str(cache_dir)
    bootstrap_module.ensure_docling_artifacts = lambda cache_dir: Path(cache_dir) / "docling_artifacts"

    original_modules = {
        "core.settings": sys.modules.get("core.settings"),
        "tools.embedding_bootstrap": sys.modules.get("tools.embedding_bootstrap"),
    }
    sys.modules["core.settings"] = settings_module
    sys.modules["tools.embedding_bootstrap"] = bootstrap_module

    try:
        spec = importlib.util.spec_from_file_location("pdf_parser_contract", ROOT / "tools" / "pdf_parser.py")
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["pdf_parser_contract"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("pdf_parser_contract", None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_extract_item_text_skips_picture_markdown_export_that_requires_doc():
    module = _load_pdf_parser_module()
    calls = {"count": 0}

    class PictureItem:
        def export_to_markdown(self, doc):
            calls["count"] += 1
            return "picture markdown"

    assert module._extract_item_text(PictureItem(), "picture") == ""
    assert calls["count"] == 0


def test_extract_item_text_uses_markdown_export_for_non_picture_items():
    module = _load_pdf_parser_module()

    class ParagraphLikeItem:
        def export_to_markdown(self):
            return "paragraph markdown"

    assert module._extract_item_text(ParagraphLikeItem(), "paragraph") == "paragraph markdown"
