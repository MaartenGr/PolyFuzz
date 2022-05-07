import numpy as np
import pandas as pd
from typing import List
import spacy

from ._utils import cosine_similarity
from ._base import BaseMatcher


class SpacyEmbeddings(BaseMatcher):
    """
    Embed words into vectors and use cosine similarity to find
    the best matches between two lists of strings

    Arguments:
        embedding_model: The Spacy model to use, this can be either a string or the model directly
        min_similarity: The minimum similarity between strings, otherwise return 0 similarity
        top_n: The number of best matches you want returned
        cosine_method: The method/package for calculating the cosine similarity.
                        Options: "sparse", "sklearn", "knn".
                        Sparse is the fastest and most memory efficient but requires a
                        package that might be difficult to install.
                        Sklearn is a bit slower than sparse and requires significantly more memory as
                        the distance matrix is not sparse
                        Knn uses 1-nearest neighbor to extract the most similar strings
                        it is significantly slower than both methods but requires little memory
        model_id: The name of the particular instance, used when comparing models

    Usage:

    ```python
    distance_model = SpacyEmbeddings("en_core_web_md", min_similarity=0.5)
    ```

    Or if you want to directly pass a Spacy model:

    ```python
    import spacy
    embedding_model = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    distance_model = SpacyEmbeddings(embedding_model, min_similarity=0.5)
    ```
    """
    def __init__(self,
                 embedding_model = "en_core_web_md",
                 min_similarity: float = 0.75,
                 top_n: int = 1,
                 cosine_method: str = "sparse",
                 model_id: str = None):
        super().__init__(model_id)
        self.type = "Embeddings"

        if isinstance(embedding_model, str):
            self.embedding_model = spacy.load(embedding_model, exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
        elif "spacy" in str(type(embedding_model)):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct Spacy model by either using a string such as 'en_core_web_md' "
                             "or create a nlp model using: `nlp = spacy.load('en_core_web_md')")

        self.min_similarity = min_similarity
        self.top_n = top_n
        self.cosine_method = cosine_method

        self.embeddings_to = None

    def match(self,
              from_list: List[str],
              to_list: List[str] = None,
              embeddings_from: np.ndarray = None,
              embeddings_to: np.ndarray = None,
              re_train: bool = True) -> pd.DataFrame:
        """ Matches the two lists of strings to each other and returns the best mapping

        Arguments:
            from_list: The list from which you want mappings
            to_list: The list where you want to map to
            embeddings_from: Embeddings you created yourself from the `from_list`
            embeddings_to: Embeddings you created yourself from the `to_list`
            re_train: Whether to re-train the model with new embeddings
                      Set this to False if you want to use this model in production

        Returns:
            matches: The best matches between the lists of strings

        Usage:

        ```python
        model = Embeddings(min_similarity=0.5)
        matches = model.match(["string_one", "string_two"],
                              ["string_three", "string_four"])
        ```
        """
        # Extract embeddings from the `from_list`
        if not isinstance(embeddings_from, np.ndarray):
            embeddings_from = self._embed(from_list)

        # Extract embeddings from the `to_list` if it exists
        if not isinstance(embeddings_to, np.ndarray):
            if not re_train:
                embeddings_to = self.embeddings_to
            elif to_list is None:
                embeddings_to = self._embed(from_list)
            else:
                embeddings_to = self._embed(to_list)

        matches = cosine_similarity(embeddings_from, embeddings_to,
                                    from_list, to_list,
                                    self.min_similarity,
                                    top_n=self.top_n,
                                    method=self.cosine_method)

        self.embeddings_to = embeddings_to

        return matches

    def _embed(self, strings: List[str]) -> np.ndarray:
        """ Create embeddings from a list of strings """
        # Extract embeddings from a transformer model
        if "transformer" in self.embedding_model.component_names:
            embeddings = []
            for doc in strings:
                try:
                    embedding = self.embedding_model(doc)._.trf_data.tensors[-1][0].tolist()
                except:
                    embedding = self.embedding_model("An empty document")._.trf_data.tensors[-1][0].tolist()
                embeddings.append(embedding)
            embeddings = np.array(embeddings)

        # Extract embeddings from a general spacy model
        else:
            embeddings = []
            for doc in strings:
                try:
                    vector = self.embedding_model(doc).vector
                except ValueError:
                    vector = self.embedding_model("An empty document").vector
                embeddings.append(vector)
            embeddings = np.array(embeddings)

        return embeddings
