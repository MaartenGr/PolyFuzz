import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from ._utils import cosine_similarity
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
        top_n: The number of matches you want returned
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
        remove_space_ngrams: Remove n-grams that contain a space

    Usage:

    ```python
    from polymatcher.models import TFIDF
    model = TFIDF(n_gram_range=(3, 3), clean_string=True)
    ```
    """
    def __init__(self,
                 n_gram_range: Tuple[int, int] = (3, 3),
                 clean_string: bool = True,
                 min_similarity: float = 0.75,
                 top_n: int = 1,
                 cosine_method: str = "sparse",
                 model_id: str = None,
                 remove_space_ngrams = True):
        super().__init__(model_id)
        self.type = "TF-IDF"
        self.n_gram_range = n_gram_range
        self.clean_string = clean_string
        self.min_similarity = min_similarity
        self.cosine_method = cosine_method
        self.top_n = top_n
        self.vectorizer = None
        self.tf_idf_to = None
        self.remove_space_ngrams = remove_space_ngrams

    def match(self,
              from_list: List[str],
              to_list: List[str] = None,
              re_train: bool = True) -> pd.DataFrame:
        """ Match two lists of strings to each other and return the most similar strings

        Arguments:
            from_list: The list from which you want mappings
            to_list: The list where you want to map to
            re_train: Whether to re-train the model with new embeddings
                      Set this to False if you want to use this model in production

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

        tf_idf_from, tf_idf_to = self._extract_tf_idf(from_list, to_list, re_train)
        matches = cosine_similarity(tf_idf_from, tf_idf_to,
                                    from_list, to_list,
                                    self.min_similarity,
                                    top_n=self.top_n,
                                    method=self.cosine_method)

        return matches

    def _extract_tf_idf(self,
                        from_list: List[str],
                        to_list: List[str] = None, 
                        re_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate distances between TF-IDF vectors of from_list and to_list """
        if to_list:
            if re_train:
                self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit(to_list + from_list)
                self.tf_idf_to = self.vectorizer.transform(to_list)
            tf_idf_from = self.vectorizer.transform(from_list)
        else:
            if re_train:
                self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit(from_list)
                self.tf_idf_to = self.vectorizer.transform(from_list)
            tf_idf_from = self.tf_idf_to

        return tf_idf_from, self.tf_idf_to

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
            if self.remove_space_ngrams:
                ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
            else:
                ngrams = [''.join(ngram) for ngram in ngrams]
            result.extend(ngrams)

        return result


def _clean_string(string: str) -> str:
    """ Only keep alphanumerical characters """
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string.lower())
    string = re.sub('\s+', ' ', string).strip()
    return string
