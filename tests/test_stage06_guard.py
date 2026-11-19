"""Stage 06 stops before writing any export when a country and day type has
between 1 and 49 household-days, since its size could be derived from the
published totals."""
import importlib.util

import pytest

from conftest import run_stage
from synthetic import prepared_data

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kneed") is None,
                                reason="kneed is not installed")


def test_small_stratum_stops_stage06(stage_dirs):
    # Luxembourg (code 9): 120 weekdays (DDV1 = 3) and 17 Sundays (DDV1 = 1)
    data = prepared_data({(0, 3): 150, (0, 1): 150, (9, 3): 120, (9, 1): 17})
    data.to_csv(stage_dirs["HETUS_PROCESSED_DIR"] / "prepared_clustering_data.csv", index=False)
    done = run_stage("06_clustering_country_day.py", stage_dirs)
    assert done.returncode != 0
    assert "('LU', 'non_working'): 17" in done.stderr
    written = [p.name for p in stage_dirs["HETUS_PROCESSED_DIR"].rglob("*.csv")]
    assert written == ["prepared_clustering_data.csv"]
