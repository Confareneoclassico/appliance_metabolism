"""A relative HETUS_* location is taken from the repository root, whichever
folder a stage is run from (L9)."""
import os
import subprocess
import sys

from conftest import PIPELINE, ROOT


def resolved(variable, value, cwd):
    env = dict(os.environ)
    env[variable] = value
    code = f"import sys; sys.path.insert(0, {str(PIPELINE)!r}); import paths; print(paths.{{}})"
    attr = {"HETUS_PROCESSED_DIR": "PROCESSED_DIR", "HETUS_DATA_DIR": "RAW_DIR"}[variable]
    out = subprocess.run([sys.executable, "-c", code.format(attr)], env=env, cwd=cwd,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_relative_locations_from_the_repository_root(tmp_path):
    sub = "processed_l9_test"
    try:
        a = resolved("HETUS_PROCESSED_DIR", sub, cwd=tmp_path)
        b = resolved("HETUS_PROCESSED_DIR", sub, cwd=PIPELINE)
        assert a == b == str(ROOT / sub)
    finally:
        for folder in sorted((ROOT / sub).rglob("*"), reverse=True):
            folder.rmdir()
        (ROOT / sub).rmdir()


def test_absolute_and_empty_values(tmp_path):
    assert resolved("HETUS_DATA_DIR", str(tmp_path), cwd=tmp_path) == str(tmp_path)
    assert resolved("HETUS_DATA_DIR", "", cwd=tmp_path) == str(ROOT / "data" / "raw")
