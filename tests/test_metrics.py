import pytest
from polyfuzz.metrics import precision_recall_curve, visualize_precision_recall
from polyfuzz.models import TFIDF
from tests.utils import get_test_strings


from_list, to_list = get_test_strings()
model = TFIDF(cosine_method="sparse")
matches = model.match(from_list, to_list)


@pytest.mark.parametrize("precision_steps", [.01, .05, .1, .2, .5])
def test_linkage(precision_steps):
    min_precisions, recall, average_precision = precision_recall_curve(matches, precision_steps=precision_steps)

    assert isinstance(min_precisions, list)
    assert isinstance(recall, list)
    assert isinstance(average_precision, list)

    assert int(1/precision_steps) + 1 == len(min_precisions)
    assert int(1 / precision_steps) + 1 == len(recall)
    assert int(1 / precision_steps) + 1 == len(average_precision)
    assert min_precisions[-1] == 1.0

    assert min_precisions[0] < min_precisions[-1]
    assert recall[0] > recall[-1]
    assert average_precision[0] < average_precision[-1]
