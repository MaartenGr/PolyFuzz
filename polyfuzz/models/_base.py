import pandas as pd
from typing import List
from abc import ABC, abstractmethod


class BaseMatcher(ABC):
    """ The abstract BaseMatching to be modelled after for string matching """

    def __init__(self, model_id: str = "Model 0"):
        self.model_id = model_id
        self.type = "Base Model"

    @abstractmethod
    def match(self,
              from_list: List[str],
              to_list: List[str] = None,
              **kwargs) -> pd.DataFrame:
        """ Make sure you follow the same argument structure:

        Arguments:
            from_list: The list from which you want mappings
            to_list: The list where you want to map to

        Returns:
            matches: The best matches between the lists of strings
                     Columns:
                        * "From"
                        * "To"
                        * "Similarity"
        """
        raise NotImplementedError()
