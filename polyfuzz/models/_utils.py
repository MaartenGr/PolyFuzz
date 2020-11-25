import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sparse_dot_topn import awesome_cossim_topn
    _HAVE_SPARSE_DOT = True
except ImportError:
    _HAVE_SPARSE_DOT = False


def extract_best_matches(from_vector: np.ndarray,
                         from_list: List[str],
                         to_vector: np.ndarray,
                         to_list: List[str],
                         min_similarity: float = 0.75,
                         method: str = "sparse") -> pd.DataFrame:
    """ Calculate similarity between two matrices

    Arguments:
        from_vector: the matrix or vector representing the embedded strings to map from
        from_list: The list from which you want mappings
        to_vector: the matrix or vector representing the embedded strings to map to
        to_list: The list where you want to map to
        min_similarity: The minimum similarity between strings, otherwise return 0 similarity
        method: The method/package for calculating the cosine similarity.
                Options: "sparse", "sklearn", "knn".
                Sparse is the fastest and most memory efficient but requires a
                package that might be difficult to install.
                Sklearn is a bit slower than sparse and requires significantly more memory as
                the distance matrix is not sparse
                Knn uses 1-nearest neighbor to extract the most similar strings
                it is significantly slower than both methods but requires little memory

    Returns:
        matches:  The best matches between the lists of strings

    Usage:

    Make sure to fill the `to_vector` and `from_vector` with vector representations
    of `to_list` and `from_list` respectively:

    ```python
    from polyfuzz.models import extract_best_matches
    matches = extract_best_matches(to_vector, from_list, from_vector, to_list, method="sparse")
    ```
    """
    # Slower but uses less memory
    if method == "knn":

        if from_list == to_list:
            knn = NearestNeighbors(n_neighbors=2, n_jobs=-1, metric='cosine').fit(from_vector)
            distances, indices = knn.kneighbors(to_vector)
            distances = distances[:, 1]
            indices = indices[:, 1]

        else:
            knn = NearestNeighbors(n_neighbors=1, n_jobs=-1, metric='cosine').fit(from_vector)
            distances, indices = knn.kneighbors(to_vector)

        similarity = [round(1 - distance, 3) for distance in distances.flatten()]

    # Fast, but does has some installation issues
    elif _HAVE_SPARSE_DOT and method == "sparse":
        if isinstance(to_vector, np.ndarray):
            to_vector = csr_matrix(to_vector)
        if isinstance(from_vector, np.ndarray):
            from_vector = csr_matrix(from_vector)

        # There is a bug with awesome_cossim_topn that when to_vector and from_vector
        # have the same shape, setting topn to 1 does not work. Apparently, you need
        # to it at least to 2 for it to work
        similarity_matrix = awesome_cossim_topn(to_vector, from_vector.T, 2, min_similarity)

        if from_list == to_list:
            similarity_matrix = similarity_matrix.tolil()
            similarity_matrix.setdiag(0.)
            similarity_matrix = similarity_matrix.tocsr()

        indices = np.array(similarity_matrix.argmax(axis=1).T).flatten()
        similarity = similarity_matrix.max(axis=1).toarray().T.flatten()

    # Faster but uses more memory
    else:
        similarity_matrix = cosine_similarity(to_vector, from_vector)

        if from_list == to_list:
            np.fill_diagonal(similarity_matrix, 0)

        indices = similarity_matrix.argmax(axis=1)
        similarity = similarity_matrix.max(axis=1)

    matches = [to_list[idx] for idx in indices.flatten()]
    matches = pd.DataFrame(np.vstack((from_list, matches, similarity)).T, columns=["From", "To", "Similarity"])
    matches.Similarity = matches.Similarity.astype(float)
    matches.loc[matches.Similarity < 0.001, "To"] = None
    return matches
