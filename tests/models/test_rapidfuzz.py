import pandas as pd
from polyfuzz.models import RapidFuzz
from tests.utils import get_test_strings
from rapidfuzz import fuzz

from_list, to_list = get_test_strings()


def test_distance():
    model = RapidFuzz()
    matches = model.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']


def test_custom_scorer():
    model = RapidFuzz(scorer=fuzz.ratio)
    matches = model.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']


def test_score_cutoff():
    model = RapidFuzz(score_cutoff=0.95)
    matches = model.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() < 0.5
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']
