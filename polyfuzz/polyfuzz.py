import logging
import pandas as pd
from typing import List, Mapping, Union, Iterable
from polyfuzz.models import TFIDF, RapidFuzz, Embeddings, BaseMatcher
from polyfuzz.metrics import precision_recall_curve, visualize_precision_recall
from polyfuzz.models.utils import cluster_mappings
from polyfuzz.utils import check_matches, create_logger

logger = create_logger()


class PolyFuzz:
    """
    PolyFuzz class for Fuzzy string matching, grouping, and evaluation.

    Arguments:
        method: the method(s) used for matching. For quick selection of models
                select one of the following:
                    * "EditDistance"
                    * "TF-IDF"
                    * "Embeddings"

                If you want more control over the models above, pass
                in a model from polyfuzz.models. For examples, see
                usage below.
        verbose: Changes the verbosity of the model, Set to True if you want
                 to track the stages of the model.

    Usage:

    For basic, out-of-the-box usage, run the code below. You can replace "TF-IDF"
    with either "EditDistance"  or "Embeddings" for quick access to these models:

    ```python
    import polyfuzz as pf
    model = pf.PolyFuzz("TF-IDF")
    ```

    If you want more control over the String Matching models, you can load
    in these models separately:

    ```python
    tfidf_matcher = TFIDF(n_gram_range=(3, 3), min_similarity=0, model_id="TF-IDF-Sklearn")
    model = pf.PolyFuzz(tfidf_matcher)
    ```

    You can also select multiple models in order to compare performance:

    ```python
    tfidf_matcher = TFIDF(n_gram_range=(3, 3), min_similarity=0, model_id="TF-IDF-Sklearn")
    edit_matcher = EditDistance(n_jobs=-1)
    model = pf.PolyFuzz([tfidf_matcher, edit_matcher])
    ```

    To use embedding models, please use Flair word embeddings:

    ```python
    from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
    fasttext_embedding = WordEmbeddings('news')
    bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')
    embedding_matcher = Embeddings([fasttext_embedding, bert_embedding ], min_similarity=0.0)
    model = pf.PolyFuzz(embedding_matcher)
    ```
    """

    def __init__(self,
                 method: Union[str,
                               BaseMatcher,
                               List[BaseMatcher]] = "TF-IDF",
                 verbose: bool = False):
        self.method = method
        self.matches = None

        # Metrics
        self.min_precisions = None
        self.recalls = None
        self.average_precisions = None

        # Cluster
        self.clusters = None
        self.cluster_mappings = None
        self.grouped_matches = None

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

    def match(self,
              from_list: List[str],
              to_list: List[str]):
        """ Match the from_list of strings to the to_list of strings with whatever models
        you have initialized

        Arguments:
            from_list: The list from which you want mappings
            to_list: The list where you want to map to

        Updates:
            self.matches: A dictionary with the matches from all models, can
                          be accessed with `model.get_all_matches` or
                          `model.get_match("TF-IDF-Matcher")`

        Usage:

        After having initialized your models, you can pass through lists of strings:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF-Matcher")
        model.match(from_list = ["string_one", "string_two"],
                    to_list = ["string_three", "string_four"])
        ```

        You can access the results matches with `model.get_all_matches` or a specific
        model with `model.get_match("TF-IDF-Matcher")` based on their model_id.
        """
        # Standard models - quick access
        if isinstance(self.method, str):
            if self.method == "TF-IDF":
                self.matches = {"TF-IDF": TFIDF(min_similarity=0).match(from_list, to_list)}
            elif self.method == "EditDistance":
                self.matches = {"EditDistance": RapidFuzz().match(from_list, to_list)}
            elif self.method == "Embeddings":
                self.matches = {"Embeddings": Embeddings(min_similarity=0).match(from_list, to_list)}
            logger.info(f"Ran matcher with model id = {self.method}")

        # Custom models
        elif isinstance(self.method, BaseMatcher):
            self.matches = {self.method.model_id: self.method.match(from_list, to_list)}
            logging.info(f"Ran matcher with model id = {self.method.model_id}")

        # Multiple custom models
        elif isinstance(self.method, Iterable):
            self._update_model_ids()
            self.matches = {}
            for model in self.method:
                self.matches[model.model_id] = model.match(from_list, to_list)
                logging.info(f"Ran matcher with model id = {model.model_id}")

        return self

    def visualize_precision_recall(self):
        """ Calculate and visualize precision-recall curves

        A minimum similarity score might be used to identify
        when a match could be considered to be correct. For example,
        we can assume that if a similarity score pass 0.95 we are
        quite confident that the matches are correct. This minimum
        similarity score can be defined as **precision** since it shows
        you how precise we believe the matches are at a minimum.

        **Recall** can then be defined as as the percentage of matches
        found at a certain minimum similarity score. A high recall means
        that for a certain minimum precision score, we find many matches.

        Usage:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF-Matcher")
        model.match(from_list = ["string_one", "string_two"],
                    to_list = ["string_three", "string_four"])
        model.visualize_precision_recall()
        ```
        """
        check_matches(self)

        self.min_precisions = {}
        self.recalls = {}
        self.average_precisions = {}

        for name, match in self.matches.items():
            min_precision, recall, average_precision = precision_recall_curve(match)
            self.min_precisions[name] = min_precision
            self.recalls[name] = recall
            self.average_precisions[name] = average_precision

        visualize_precision_recall(self.matches, self.min_precisions, self.recalls)

    def group(self, minimum_similarity: float = 0.8):
        """ From the matches, group the `To` matches together using single linkage

         Arguments:
             minimum_similarity: the minimum similarity between strings before they are grouped
                                 in a single linkage fashion

         Updates:
            self.matches: Adds a column `Group` that is the grouped version of the `To` column
         """
        check_matches(self)

        self.clusters = {}
        self.cluster_mappings = {}

        for name, match in self.matches.items():
            strings = list(self.matches[name].To.dropna().unique())
            tfidf, _ = TFIDF(n_gram_range=(3, 3))._extract_tf_idf(strings, None)
            clusters, cluster_id_map, cluster_name_map = cluster_mappings(tfidf, strings, minimum_similarity)
            self._map_groups(name, cluster_name_map)
            self.clusters[name] = clusters
            self.cluster_mappings[name] = cluster_id_map

    def get_all_model_ids(self) -> Union[str, List[str], None]:
        """ Get all model ids for easier access """
        check_matches(self)

        if isinstance(self.method, str):
            return self.method
        elif isinstance(self.method, Iterable):
            return [model.model_id for model in self.method]
        return None

    def get_all_matches(self) -> Union[pd.DataFrame,
                                       Mapping[str, pd.DataFrame]]:
        """ Returns the matches from all models """
        check_matches(self)
        return self.matches

    def get_matches(self, name: str = None) -> pd.DataFrame:
        """ Get the matches from a single model """
        check_matches(self)

        if len(self.matches) == 1:
            return list(self.matches.values())[0]

        elif len(self.matches) > 1 and not name:
            raise ValueError(f"Please use the parameter 'name' with one of the following values: "
                             f"{self.get_all_model_ids()}")

        return self.matches[name]

    def get_clusters(self, name: str = None) -> Mapping[str, List[str]]:
        """ Get the groupings/clusters from a single model """
        check_matches(self)

        if len(self.matches) == 1:
            return list(self.clusters.values())[0]

        elif len(self.matches) > 1 and not name:
            raise ValueError(f"Please use the parameter 'name' with one of the following values: "
                             f"{self.get_all_model_ids()}")

        return self.clusters[name]

    def get_cluster_mappings(self, name: str = None) -> Mapping[str, int]:
        """ Get the mappings from the `To` column to its respective column """
        check_matches(self)

        if len(self.matches) == 1:
            return list(self.cluster_mappings.values())[0]

        elif len(self.matches) > 1 and not name:
            raise ValueError(f"Please use the parameter 'name' with one of the following values: "
                             f"{self.get_all_model_ids()}")

        return self.cluster_mappings[name]

    def _map_groups(self, name: str, cluster_name_map: Mapping[str, str]):
        """ Map the 'to' list to groups """
        df = self.matches[name]
        df["Group"] = df['To'].map(cluster_name_map).fillna(df['To'])

        # Fix that some mappings from "From" end up in "Group"
        df.loc[(df.From != df.To) &
               (df.From == df.Group), "Group"] = df.loc[(df.From != df.To) &
                                                        (df.From == df.Group), "To"]
        self.matches[name] = df

    def _update_model_ids(self):
        """ Update model ids such that there is no overlap between ids """
        # Give models a model_id if it didn't already exist
        for index, model in enumerate(self.method):
            if not model.model_id:
                model.model_id = f"Model {index}"

        # Update duplicate names
        model_ids = [model.model_id for model in self.method]
        if len(set(model_ids)) != len(model_ids):
            for index, model in enumerate(self.method):
                model.model_id = f"Model {index}"
