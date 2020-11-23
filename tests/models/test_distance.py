import pandas as pd
from polyfuzz.models import EditDistance
from tests.utils import get_test_strings
from rapidfuzz import fuzz

from_list, to_list = get_test_strings()


def test_distance():
    matcher = EditDistance()
    matches = matcher.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']


def test_custom_scorer():
    matcher = EditDistance(scorer=fuzz.ratio)
    matches = matcher.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']


def test_no_normalization():
    matcher = EditDistance(normalize=False)
    matches = matcher.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 50
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']
