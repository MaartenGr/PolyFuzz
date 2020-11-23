import pandas as pd
from typing import List, Mapping, Tuple


def single_linkage(matches: pd.DataFrame,
                   min_similarity: float = 0.8) -> Tuple[Mapping[int, List[str]],
                                                         Mapping[str, int],
                                                         Mapping[str, str]]:
    """ Single linkage clustering from column 'From' to column 'To'

    `matches` contains three columns: *From*, *To*, and *Similarity* where
    *Similarity* is already the minimum similarity score and thus no checking
    for minimum similarity is necessary.

    Arguments:
        matches: contains the columns *From*, *To*, and *Similarity* used for creating groups
        min_similarity: minimum similarity between strings before they can be merged into a group

    Returns:
        clusters: The populated clusters
        cluster_mapping: The mapping from a string to a cluster
        cluster_name_map: The mapping from a string to the representative string
                          in its respective cluster
    """
    matches = matches.loc[matches.Similarity > min_similarity, :]

    cluster_mapping = {}
    cluster_id = 0

    for row in matches.itertuples():

        # If from string has not already been mapped
        if not cluster_mapping.get(row.From):

            # If the to string has not already been mapped
            if not cluster_mapping.get(row.To):
                cluster_mapping[row.To] = cluster_id
                cluster_mapping[row.From] = cluster_id
                cluster_id += 1

            # If the to string has already been mapped
            else:
                cluster_mapping[row.From] = cluster_mapping.get(row.To)

    # Populate the clusters
    clusters = {}
    for key, value in cluster_mapping.items():
        clusters.setdefault(value, [])
        clusters[value].append(key)

    cluster_name_map = {key: clusters.get(value)[0] for key, value in cluster_mapping.items()}

    return clusters, cluster_mapping, cluster_name_map
