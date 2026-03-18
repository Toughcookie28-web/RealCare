from pathlib import Path


def test_all_runtime_install_paths_reference_the_same_profile():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/eval.yml").read_text(encoding="utf-8")
    render = Path("render.yaml").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python3 -m pip install --no-cache-dir ." in dockerfile
    assert 'python3 -m pip install ".[dev]"' in workflow
    assert "python3 -m pip install ." in workflow
    assert 'python3 -m pip install ".[eval,ingest]"' in workflow
    assert "buildCommand: python3 -m pip install ." in render
    assert "python3 -m pip install ." in readme
    assert 'python3 -m pip install ".[eval,ingest]"' in readme
