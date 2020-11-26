import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from polyfuzz.models._utils import extract_best_matches
from ._base import BaseMatcher


class TFIDF(BaseMatcher):
    """
    A character based n-gram TF-IDF to approximate edit distance

    We turn a string into, typically of length 3, n-grams. For example,
    using 3-grams of the "hotel" we get ['hot', 'ote', 'tel']. These are then
    used as input for a TfidfVectorizer in order to create a vector for each
    word. Then, we simply apply cosine similarity through k-NN

    Arguments:
        n_gram_range: The n_gram_range on a character-level
        clean_string: Whether to clean the string such that only alphanumerical characters are kept
        min_similarity: The minimum similarity between strings, otherwise return 0 similarity
        cosine_method: The method/package for calculating the cosine similarity.
                        Options:
                            * sparse
                            * sklearn
                            * knn

                        sparse is the fastest and most memory efficient but requires a
                        package that might be difficult to install

                        sklearn is a bit slower than sparse and requires significantly more memory as
                        the distance matrix is not sparse

                        knn uses 1-nearest neighbor to extract the most similar strings
                        it is significantly slower than both methods but requires little memory
        model_id: The name of the particular instance, used when comparing models

    Usage:

    ```python
    from polymatcher.models import TFIDF
    model = TFIDF(n_gram_range=(3, 3), clean_string=True, use_knn=False)
    ```
    """
    def __init__(self,
                 n_gram_range: Tuple[int, int] = (3, 3),
                 clean_string: bool = True,
                 min_similarity: float = 0.75,
                 cosine_method: str = "sparse",
                 model_id: str = None):
        super().__init__(model_id)
        self.type = "TF-IDF"
        self.n_gram_range = n_gram_range
        self.clean_string = clean_string
        self.min_similarity = min_similarity
        self.cosine_method = cosine_method

    def match(self,
              from_list: List[str],
              to_list: List[str]) -> pd.DataFrame:
        """ Match two lists of strings to each other and return the most similar strings

        Arguments:
            from_list: The list from which you want mappings
            to_list: The list where you want to map to

        Returns:
            matches: The best matches between the lists of strings

        Usage:

        ```python
        from polymatcher.models import TFIDF
        model = TFIDF()
        matches = model.match(["string_one", "string_two"],
                              ["string_three", "string_four"])
        ```
        """

        tf_idf_from, tf_idf_to = self._extract_tf_idf(from_list, to_list)
        matches = extract_best_matches(tf_idf_from, from_list, tf_idf_to, to_list,
                                       self.min_similarity, self.cosine_method)

        return matches

    def _extract_tf_idf(self,
                        from_list: List[str],
                        to_list: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate distances between TF-IDF vectors of from_list and to_list """
        if to_list:
            vectorizer = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit(to_list + from_list)
            tf_idf_to = vectorizer.transform(to_list)
            tf_idf_from = vectorizer.transform(from_list)
        else:
            tf_idf_to = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit_transform(from_list)
            tf_idf_from = None

        return tf_idf_to, tf_idf_from

    def _create_ngrams(self, string: str) -> List[str]:
        """ Create n_grams from a string

        Steps:
            * Extract character-level ngrams with `self.n_gram_range` (both ends inclusive)
            * Remove n-grams that have a whitespace in them
        """
        if self.clean_string:
            string = _clean_string(string)

        result = []
        for n in range(self.n_gram_range[0], self.n_gram_range[1]+1):
            ngrams = zip(*[string[i:] for i in range(n)])
            ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
            result.extend(ngrams)

        return result


def _clean_string(string: str) -> str:
    """ Only keep alphanumerical characters """
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string.lower())
    string = re.sub('\s+', ' ', string).strip()
    return string
