import pytest
import numpy as np
import pandas as pd
from polyfuzz.models import Embeddings
from tests.utils import get_test_strings

from_list, to_list = get_test_strings()

with open('tests/from_list.npy', 'rb') as f:
    from_vector = np.load(f)

with open('tests/to_list.npy', 'rb') as f:
    to_vector = np.load(f)


@pytest.mark.parametrize("method", ["sparse", "knn", "sklearn"])
def test_embeddings(method):
    model = Embeddings(embedding_method="None", min_similarity=0, cosine_method=method)
    matches = model.match(from_list, to_list, from_vector, to_vector)

    assert isinstance(matches, pd.DataFrame)
    assert matches.Similarity.mean() > 0.0
    assert len(matches) == 6
    assert list(matches.columns) == ['From', 'To', 'Similarity']
