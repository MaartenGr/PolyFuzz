import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz
from typing import List, Tuple, Callable, Union
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from ._base import BaseMatcher


class RapidFuzz(BaseMatcher):
    """
    Calculate the Edit Distance between lists of strings using RapidFuzz's process function

    We are using RapidFuzz instead of FuzzyWuzzy since it is much faster
    and does not require the more restrictive GPL license

    Arguments:
        n_jobs: Nr of parallel processes, use -1 to use all cores
        score_cutoff: The minimum similarity for which to return a good match.
                      Should be between 0 and 1.
        scorer: The scorer function to be used to calculate the edit distance
                Options:
                    * fuzz.ratio
                    * fuzz.partial_ratio
                    * fuzz.token_sort_ratio
                    * fuzz.partial_token_sort_ratio
                    * fuzz.token_set_ratio
                    * fuzz.partial_token_set_ratio
                    * fuzz.token_ratio
                    * fuzz.partial_token_ratio
                    * fuzz.WRation
                    * fuzz.QRatio
                See https://maxbachmann.github.io/rapidfuzz/usage/fuzz/ for an extensive
                description of the scoring methods.
        model_id: The name of the particular instance, used when comparing models

    Usage:

    ```python
    from rapidfuzz import fuzz
    model = RapidFuzz(n_jobs=-1, score_cutoff=0.5, scorer=fuzz.WRatio)
    ```
    """
    def __init__(self,
                 n_jobs: int = 1,
                 score_cutoff: float = 0,
                 scorer: Callable = fuzz.WRatio,
                 model_id: str = None):
        super().__init__(model_id)
        self.type = "EditDistance"
        self.score_cutoff = score_cutoff * 100
        self.scorer = scorer
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
        model = RapidFuzz(n_jobs=-1, score_cutoff=0.5, scorer=fuzz.WRatio)
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
        return matches

    def _calculate_edit_distance(self,
                                 from_string: str,
                                 to_list: List[str]) -> Tuple[str, Union[str, None], float]:
        """ Calculate the edit distance between a string and a list """
        if self.equal_lists:
            to_list.remove(from_string)

        match = process.extractOne(from_string, to_list,
                                   score_cutoff=self.score_cutoff,
                                   scorer=self.scorer)

        if match:
            return from_string, match[0], match[1] / 100
        else:
            return from_string, None, 0.
