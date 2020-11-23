import pytest
import pandas as pd
from polyfuzz.models import TFIDF
from tests.utils import get_test_strings

from_list, to_list = get_test_strings()


@pytest.mark.parametrize("method", ["sparse", "knn", "sklearn"])
def test_distance(method):
    model = TFIDF(cosine_method=method)
    matches = model.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']


@pytest.mark.parametrize("n_gram_low, n_gram_high", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)])
def test_ngrams(n_gram_low, n_gram_high):
    model = TFIDF(n_gram_range=(n_gram_low, n_gram_high))
    matches = model.match(from_list, to_list)

    assert isinstance(matches, pd.DataFrame)
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']
