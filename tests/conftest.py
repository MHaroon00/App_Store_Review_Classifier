import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labeler.data import read_table  # noqa: E402
from labeler.prompts import LabelSet  # noqa: E402


@pytest.fixture
def sample_df():
    p = ROOT / "sample_data" / "reviews_sample.csv"
    return read_table(p.read_bytes(), p.name)


@pytest.fixture
def general_labels():
    return LabelSet.from_dict(json.loads((ROOT / "sample_data" / "labels_general.json").read_text()))


@pytest.fixture
def nielsen_labels():
    return LabelSet.from_dict(json.loads((ROOT / "sample_data" / "labels_usability_nielsen.json").read_text()))
