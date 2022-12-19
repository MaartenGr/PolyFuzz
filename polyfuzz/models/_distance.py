import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from typing import List, Tuple, Callable
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from ._base import BaseMatcher


class EditDistance(BaseMatcher):
    """
    Calculate the Edit Distance between lists of strings using any distance/similarity based scorer

    Arguments:
        n_jobs: Nr of parallel processes, use -1 to use all cores
        scorer: The scorer function to be used to calculate the edit distance.
                This function should give back a float between 0 and 1, and work as follows:
                    scorer("string_one", "string_two")
        model_id: The name of the particular instance, used when comparing models

    Usage:

    ```python
    from rapidfuzz import fuzz
    model = EditDistance(n_jobs=-1, scorer=fuzz.WRatio)
    ```
    """
    def __init__(self,
                 n_jobs: int = 1,
                 scorer: Callable = fuzz.ratio,
                 model_id: str = None,
                 normalize: bool = True):
        super().__init__(model_id)
        self.type = "EditDistance"
        self.scorer = scorer
        self.normalize = normalize
        self.equal_lists = False

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    def match(self,
              from_list: List[str],
              to_list: List[str] = None,
              **kwargs) -> pd.DataFrame:
        """ Calculate the edit distances between two list of strings
        by parallelizing the calculation and passing the lists in
        batches.

        Arguments:
            from_list: The list from which you want mappings
            to_list: The list where you want to map to

        Returns:
            matches: The best matches between the lists of strings

        Usage:

        ```python
        from rapidfuzz import fuzz
        model = EditDistance(n_jobs=-1, score_cutoff=0.5, scorer=fuzz.WRatio)
        matches = model.match(["string_one", "string_two"],
                              ["string_three", "string_four"])
        ```
        """
        if to_list is None:
            self.equal_lists = True
            expected_iterations = int(len(from_list)/2)
            to_list = from_list.copy()
        else:
            expected_iterations = len(from_list)

        matches = Parallel(n_jobs=self.n_jobs)(delayed(self._calculate_edit_distance)
                                               (from_string, to_list)
                                               for from_string in tqdm(from_list, total=expected_iterations,
                                                                       disable=True))
        matches = pd.DataFrame(matches, columns=['From', "To", "Similarity"])

        if self.normalize:
            matches["Similarity"] = (matches["Similarity"] -
                                     matches["Similarity"].min()) / (matches["Similarity"].max() -
                                                                     matches["Similarity"].min())
        return matches

    def _calculate_edit_distance(self,
                                 from_string: str,
                                 to_list: List[str]) -> Tuple[str, str, float]:
        """ Calculate the edit distance between a string and a list """
        list_to_match = to_list.copy()
        
        if self.equal_lists:
            list_to_match.remove(from_string)

        matches = [self.scorer(from_string, to_string) for to_string in list_to_match]
        index = np.argmax(matches)
        value = np.max(matches)

        return from_string, list_to_match[index], value
