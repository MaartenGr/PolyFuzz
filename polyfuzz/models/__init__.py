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

try:
    from._sbert import SentenceEmbeddings
except ModuleNotFoundError as e:
    SentenceEmbeddings = NotInstalled("SentenceTransformers", "sbert")

try:
    from._gensim import GensimEmbeddings
except ModuleNotFoundError as e:
    GensimEmbeddings = NotInstalled("Gensim", "gensim")

try:
    from._spacy import SpacyEmbeddings
except ModuleNotFoundError as e:
    SpacyEmbeddings = NotInstalled("Spacy", "spacy")

try:
    from._use import USEEmbeddings
except ModuleNotFoundError as e:
    USEEmbeddings = NotInstalled("USE", "use")


__all__ = [
    "BaseMatcher",
    "EditDistance",
    "Embeddings",
    "SentenceEmbeddings",
    "GensimEmbeddings",
    "SpacyEmbeddings",
    "USEEmbeddings",
    "RapidFuzz",
    "TFIDF",
    "cosine_similarity"
]
