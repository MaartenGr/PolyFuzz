import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as scikit_cosine_similarity

try:
    from sparse_dot_topn import awesome_cossim_topn
    _HAVE_SPARSE_DOT = True
except ImportError:
    _HAVE_SPARSE_DOT = False


def cosine_similarity(from_vector: np.ndarray,
                      to_vector: np.ndarray,
                      from_list: List[str],
                      to_list: List[str],
                      min_similarity: float = 0.75,
                      top_n: int = 1,
                      method: str = "sparse") -> pd.DataFrame:
    """ Calculate similarity between two matrices/vectors and return best matches

    Arguments:
        from_vector: the matrix or vector representing the embedded strings to map from
        to_vector: the matrix or vector representing the embedded strings to map to
        from_list: The list from which you want mappings
        to_list: The list where you want to map to
        min_similarity: The minimum similarity between strings, otherwise return 0 similarity
        top_n: The number of best matches you want returned
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
    indices, similarity = extract_best_matches(from_vector, to_vector, method="sparse")
    ```
    """
    if to_list is not None:
        if top_n > len(set(to_list)):
            top_n = len(set(to_list))
    
    # Slower but uses less memory
    if method == "knn":

        if to_list is None:
            knn = NearestNeighbors(n_neighbors=top_n+1, n_jobs=-1, metric='cosine').fit(to_vector)
            distances, indices = knn.kneighbors(from_vector)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            knn = NearestNeighbors(n_neighbors=top_n, n_jobs=-1, metric='cosine').fit(to_vector)
            distances, indices = knn.kneighbors(from_vector)

        similarities = [np.round(1 - distances[:, i], 3) for i in range(distances.shape[1])]

    # Fast, but does has some installation issues
    elif _HAVE_SPARSE_DOT and method == "sparse":
        if isinstance(to_vector, np.ndarray):
            to_vector = csr_matrix(to_vector)
        if isinstance(from_vector, np.ndarray):
            from_vector = csr_matrix(from_vector)

        # There is a bug with awesome_cossim_topn that when to_vector and from_vector
        # have the same shape, setting topn to 1 does not work. Apparently, you need
        # to it at least to 2 for it to work
        similarity_matrix = awesome_cossim_topn(from_vector, to_vector.T, top_n+1, min_similarity)

        if to_list is None:
            similarity_matrix = similarity_matrix.tolil()
            similarity_matrix.setdiag(0.)
            similarity_matrix = similarity_matrix.tocsr()

        indices = _top_n_idx_sparse(similarity_matrix, top_n)
        similarities = _top_n_similarities_sparse(similarity_matrix, indices)
        indices = np.array(np.nan_to_num(np.array(indices, dtype=np.float32), nan=0), dtype=np.int32)

    # Faster than knn and slower than sparse but uses more memory
    else:
        similarity_matrix = scikit_cosine_similarity(from_vector, to_vector)

        if to_list is None:
            np.fill_diagonal(similarity_matrix, 0)

        indices = np.flip(np.argsort(similarity_matrix, axis=-1), axis=1)[:, :top_n]
        similarities = np.flip(np.sort(similarity_matrix, axis=-1), axis=1)[:, :top_n]
        similarities = [np.round(similarities[:, i], 3) for i in range(similarities.shape[1])]

    # Convert results to df
    if to_list is None:
        to_list = from_list.copy()
        
    columns = (["From"] +
               ["To" if i == 0 else f"To_{i+1}" for i in range(top_n)] +
               ["Similarity" if i ==0 else f"Similarity_{i+1}" for i in range(top_n)])
    matches = [[to_list[idx] for idx in indices[:, i]] for i in range(indices.shape[1])]
    matches = pd.DataFrame(np.vstack(([from_list], matches, similarities)).T, columns = columns)

    # Update column order
    columns = [["From", "To", "Similarity"]] + [[f"To_{i+2}", f"Similarity_{i+2}"] for i in range((top_n-1))]
    matches = matches.loc[:, [title for column in columns for title in column]]

    # Update types
    for column in matches.columns:
        if "Similarity" in column:
            matches[column] = matches[column].astype(float)
            matches.loc[matches[column] < 0.001, column] = float(0)
            matches.loc[matches[column] < 0.001, column.replace("Similarity", "To")] = None

    return matches


def _top_n_idx_sparse(matrix, n):
    """ Return index of top n values in each row of a sparse matrix """
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = list(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])[::-1]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        top_n_idx.append(values)
    return np.array(top_n_idx)


def _top_n_similarities_sparse(matrix, indices):
    """ Return similarity scores of top n values in each row of a sparse matrix """
    similarity_scores = []
    for row, values in enumerate(indices):
        scores = [round(matrix[row, value], 3) if value is not None else 0 for value in values]
        similarity_scores.append(scores)
    similarity_scores = np.array(similarity_scores).T
    return similarity_scores
