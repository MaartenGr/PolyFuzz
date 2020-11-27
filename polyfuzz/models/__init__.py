from ._base import BaseMatcher
from ._distance import EditDistance
from ._rapidfuzz import RapidFuzz
from ._tfidf import TFIDF
from ._utils import cosine_similarity

from polyfuzz.error import NotInstalled

try:
    from ._embeddings import Embeddings
except ModuleNotFoundError as e:
    Embeddings = NotInstalled("Flair and Huggingface Transformer Models", "flair")

__all__ = [
    "BaseMatcher",
    "EditDistance",
    "Embeddings",
    "RapidFuzz",
    "TFIDF",
    "cosine_similarity"
]
