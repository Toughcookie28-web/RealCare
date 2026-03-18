import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _get_function(tree: ast.AST, name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            if isinstance(item.func, ast.Name):
                names.add(item.func.id)
            elif isinstance(item.func, ast.Attribute):
                names.add(item.func.attr)
    return names


def _attribute_names(node: ast.AST) -> set[str]:
    return {
        item.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
    }


def test_app_startup_does_not_trigger_pdf_ingest():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    lifespan = _get_function(tree, "lifespan")

    assert "init_database" in _called_names(lifespan)
    assert "ingest_pdf_to_vector_store" not in _called_names(lifespan)
    assert "auto_ingest_on_startup" not in _attribute_names(lifespan)


def test_reindex_script_remains_explicit_ingest_entrypoint():
    tree = ast.parse((ROOT / "scripts/reindex_pdf.py").read_text(encoding="utf-8"))

    assert "init_database" in _called_names(tree)
    assert "ingest_pdf_to_vector_store" in _called_names(tree)
