import pytest
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, RapidFuzz, BaseMatcher

from tests.utils import get_test_strings

from_list, to_list = get_test_strings()


class MyModel(BaseMatcher):
    def match(self, from_list, to_list):
        # Calculate distances
        matches = [[fuzz.ratio(from_string, to_string) / 100 for to_string in to_list] for from_string in from_list]

        # Get best matches
        mappings = [to_list[index] for index in np.argmax(matches, axis=1)]
        scores = np.max(matches, axis=1)

        # Prepare dataframe
        matches = pd.DataFrame({'From': from_list, 'To': mappings, 'Similarity': scores})
        return matches


@pytest.mark.parametrize("method", ["EditDistance", "TF-IDF"])
def test_base_model(method):
    model = PolyFuzz(method).match(from_list, to_list)
    matches = model.get_matches()

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.3
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']


@pytest.mark.parametrize("method", ["EditDistance", "TF-IDF"])
def test_grouper(method):
    model = PolyFuzz(method).match(from_list, to_list)
    model.group(link_min_similarity=0.75)
    matches = model.get_matches()

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.3
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity', 'Group']

    assert model.get_clusters() == {1: ['apples', 'apple']}
    assert model.get_cluster_mappings() == {'apples': 1, 'apple': 1}


@pytest.mark.parametrize("method", ["Unknown Model"])
def test_wrongbase_model(method):
    with pytest.raises(ValueError):
        model = PolyFuzz(method).match(from_list, to_list)


def test_multiple_models():
    tfidf_matcher = TFIDF(n_gram_range=(3, 3), min_similarity=0, model_id="TF-IDF")
    tfidf_large_matcher = TFIDF(n_gram_range=(3, 6), min_similarity=0)
    base_edit_matcher = EditDistance(n_jobs=1)
    ratio_matcher = EditDistance(n_jobs=1, scorer=fuzz.ratio)
    rapidfuzz_matcher = RapidFuzz(n_jobs=1)
    matchers = [tfidf_matcher, tfidf_large_matcher, base_edit_matcher, ratio_matcher , rapidfuzz_matcher]

    model = PolyFuzz(matchers).match(from_list, to_list)

    # Test if correct matches are found
    for model_id in model.get_ids():
        assert model_id in model.get_matches().keys()
        assert isinstance(model.get_matches(model_id), pd.DataFrame)
    assert len(model.get_matches()) == len(matchers)

    # Test if error is raised when accessing clusters before creating them
    with pytest.raises(ValueError):
        model.get_clusters()

    with pytest.raises(ValueError):
        model.get_cluster_mappings()

    # Test if groupings are found
    model.group()
    for model_id in model.get_ids():
        assert model_id in model.get_cluster_mappings().keys()
    assert len(model.get_cluster_mappings()) == len(matchers)


def test_custom_model():
    custom_matcher = MyModel()
    model = PolyFuzz(custom_matcher).match(from_list, to_list)
    matches = model.get_matches()
    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']

