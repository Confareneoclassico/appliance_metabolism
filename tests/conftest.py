"""Test set-up: the stages are imported from pipeline/, and every location of
paths.py points to a temporary folder, so that no test reads or writes the
folders of a real run."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
_TMP = Path(tempfile.mkdtemp(prefix="hetus_tests_"))
for var, sub in (("HETUS_DATA_DIR", "raw"), ("HETUS_PROCESSED_DIR", "processed"),
                 ("HETUS_FIGURES_DIR", "figures")):
    os.environ[var] = str(_TMP / sub)
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_stage(script, env_dirs, extra_env=None, timeout=600):
    """Run one stage as a script with its locations in `env_dirs` (a dict of
    HETUS_* variables) and return the completed process."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_dirs.items()})
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(PIPELINE / script)], env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def stage_dirs(tmp_path):
    dirs = {"HETUS_DATA_DIR": tmp_path / "raw", "HETUS_PROCESSED_DIR": tmp_path / "processed",
            "HETUS_FIGURES_DIR": tmp_path / "figures"}
    for d in dirs.values():
        d.mkdir()
    return dirs
