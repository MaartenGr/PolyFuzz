import joblib
import logging
import pandas as pd
from typing import List, Mapping, Union, Iterable

from polyfuzz.linkage import single_linkage
from polyfuzz.utils import check_matches, check_grouped, create_logger
from polyfuzz.models import TFIDF, RapidFuzz, Embeddings, BaseMatcher
from polyfuzz.metrics import precision_recall_curve, visualize_precision_recall



class PolyFuzz:
    """
    PolyFuzz class for Fuzzy string matching, grouping, and evaluation.

    Arguments:
        method: the method(s) used for matching. For quick selection of models
                select one of the following: "EditDistance", "TF-IDF" or "Embeddings".
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
    tfidf = TFIDF(n_gram_range=(3, 3), min_similarity=0, model_id="TF-IDF-Sklearn")
    model = pf.PolyFuzz(tfidf)
    ```

    You can also select multiple models in order to compare performance:

    ```python
    tfidf = TFIDF(n_gram_range=(3, 3), min_similarity=0, model_id="TF-IDF-Sklearn")
    edit = EditDistance(n_jobs=-1)
    model = pf.PolyFuzz([tfidf, edit])
    ```

    You can use embedding model, like Flair:

    ```python
    from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
    fasttext_embedding = WordEmbeddings('news')
    bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')
    embedding = Embeddings([fasttext_embedding, bert_embedding ], min_similarity=0.0)
    model = pf.PolyFuzz(embedding)
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

        # Logger
        self.logger = create_logger()
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def match(self,
              from_list: List[str],
              to_list: List[str] = None,
              top_n: int = 1):
        """ Match the from_list of strings to the to_list of strings with whatever models
        you have initialized

        Arguments:
            from_list: The list from which you want mappings.
                       If you want to map items within a list, and not map the 
                       items to themselves, you can supply only the `from_list` and 
                       ignore the `to_list`. 
            to_list: The list where you want to map to
            top_n: The number of matches you want returned. This is currently only implemented
                   for `polyfuzz.models.TFIDF` and `polyfuzz.models.Embeddings` as they
                   can computationally handle more comparisons.

        Updates:
            self.matches: A dictionary with the matches from all models, can
                          be accessed with `model.get_all_matches` or
                          `model.get_match("TF-IDF")`

        Usage:

        After having initialized your models, you can pass through lists of strings:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        model.match(from_list = ["string_one", "string_two"],
                    to_list = ["string_three", "string_four"])
        ```

        You can access the results matches with `model.get_all_matches` or a specific
        model with `model.get_match("TF-IDF")` based on their model_id.
        """
        # Standard models - quick access
        if isinstance(self.method, str):
            if self.method in ["TF-IDF", "TFIDF"]:
                self.method = TFIDF(min_similarity=0, top_n=top_n)
                self.matches = {"TF-IDF": self.method.match(from_list, to_list)}
            elif self.method in ["EditDistance", "Edit Distance"]:
                self.method = RapidFuzz()
                self.matches = {"EditDistance": self.method.match(from_list, to_list)}
            elif self.method in ["Embeddings", "Embedding"]:
                self.method = Embeddings(min_similarity=0, top_n=top_n)
                self.matches = {"Embeddings": self.method.match(from_list, to_list)}
            else:
                raise ValueError("Please instantiate the model with one of the following methods: \n"
                                 "* 'TF-IDF'\n"
                                 "* 'EditDistance'\n"
                                 "* 'Embeddings'\n")
            self.logger.info(f"Ran model with model id = {self.method}")

        # Custom models
        elif isinstance(self.method, BaseMatcher):
            self.matches = {self.method.model_id: self.method.match(from_list, to_list)}
            self.logger.info(f"Ran model with model id = {self.method.model_id}")

        # Multiple custom models
        elif isinstance(self.method, Iterable):
            self._update_model_ids()
            self.matches = {}
            for model in self.method:
                self.matches[model.model_id] = model.match(from_list, to_list)
                self.logger.info(f"Ran model with model id = {model.model_id}")

        return self

    def fit(self, 
            from_list: List[str],
            to_list: List[str] = None):
        """ Fit one or model distance models on `from_list` if no `to_list` is given 
        or fit them on `to_list` if both `from_list` and `to_list` are given. 
        
        Typically, the `to_list` will be tracked as the list that we want to transform
        our `from_list` to. In other words, it is the golden list of words that we 
        want the words in the `from_list` mapped to. 

        However, you can also choose a single `from_list` and leave `to_list` empty
        to map all words from within `from_list` to each other. Then, `from_list` 
        will be tracked instead as the golden list of words. 

        Thus, if you want to train on a single list instead, use only `from_list` 
        and keep `to_list` empty. 
        
        Arguments:
            from_list: The list from which you want mappings.
                       If you want to map items within a list, and not map the 
                       items to themselves, you can supply only the `from_list` and 
                       ignore the `to_list`. 
            to_list: The list where you want to map to

        Usage:

        After having initialized your models, you can pass through lists of strings:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        model.fit(from_list = ["string_one", "string_two"],
                  to_list = ["string_three", "string_four"])
        ```

        Now, whenever you apply `.transform(new_list)`, the `new_list` will be mapped 
        to the words in `to_list`.

        You can also fit on a single list of words:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        model.fit(["string_three", "string_four"])
        ```
        """
        self.match(from_list, to_list)
        if to_list is not None:
            self.to_list = to_list
        else:
            self.to_list = from_list
        return self

    def transform(self, from_list: List[str]) -> Mapping[str, pd.DataFrame]:
        """ After fitting your model, match all words in `from_list` 
        to the words that were fitted on previously. 
        
        Arguments:
            from_list: The list from which you want mappings.
        
        Usage:

        After having initialized your models, you can pass through lists of strings:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        model.fit(["input_string_1", "input_string2"])
        ```

        Then, you can transform and normalize new strings:

        ```python
        results = model.transform(["input_string_1", "input_string2"])
        ```
        """
        all_matches = {}

        if isinstance(self.method, BaseMatcher): 
            matches = self.method.match(from_list, self.to_list, re_train=False)
            all_matches[self.method.type] = matches

        elif isinstance(self.method, Iterable):
            for model in self.method:
                all_matches[model.type] = model.match(from_list, self.to_list, re_train=False)

        return all_matches

    def fit_transform(self, 
                      from_list: List[str], 
                      to_list: List[str] = None) -> Mapping[str, pd.DataFrame]:
        """ Fit and transform lists of words on one or more distance models.
        
        Typically, the `to_list` will be tracked as the list that we want to transform
        our `from_list` to. In other words, it is the golden list of words that we 
        want the words in the `from_list` mapped to. 

        However, you can also choose a single `from_list` and leave `to_list` empty
        to map all words from within `from_list` to each other. Then, `from_list` 
        will be tracked instead as the golden list of words. 
         
        Arguments:
            from_list: The list from which you want mappings.
                       If you want to map items within a list, and not map the 
                       items to themselves, you can supply only the `from_list` and 
                       ignore the `to_list`. 
            to_list: The list where you want to map to

        Usage:

        After having initialized your models, you can pass through lists of strings:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        results = model.fit_transform(from_list = ["string_one", "string_two"],
                                      to_list = ["string_three", "string_four"])
        ```

        You can also fit and transform a single list of words:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        results = model.fit_transform(["string_three", "string_four"])
        ```
        """
        self.fit(from_list, to_list)
        return self.transform(from_list)

    def visualize_precision_recall(self,
                                   kde: bool = False,
                                   save_path: str = None
                                   ):
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

        Arguments:
            kde: whether to also visualize the kde plot
            save_path: the path to save the resulting image to

        Usage:

        ```python
        import polyfuzz as pf
        model = pf.PolyFuzz("TF-IDF", model_id="TF-IDF")
        model.match(from_list = ["string_one", "string_two"],
                    to_list = ["string_three", "string_four"])
        model.visualize_precision_recall(save_path="results.png")
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

        visualize_precision_recall(self.matches, self.min_precisions, self.recalls, kde, save_path)

    def group(self,
              model: Union[str, BaseMatcher] = None,
              link_min_similarity: float = 0.75,
              group_all_strings: bool = False):
        """ From the matches, group the `To` matches together using single linkage

         Arguments:
             model: you can choose one of the models in `polyfuzz.models` to be used as a grouper
             link_min_similarity: the minimum similarity between strings before they are grouped
                                  in a single linkage fashion
             group_all_strings: if you want to compare a list of strings with itself and then cluster
                                those strings, set this to True. Otherwise, only the strings that
                                were mapped To are clustered.

         Updates:
            self.matches: Adds a column `Group` that is the grouped version of the `To` column
         """
        check_matches(self)
        self.clusters = {}
        self.cluster_mappings = {}

        # Standard models - quick access
        if isinstance(model, str):
            if model in ["TF-IDF", "TFIDF"]:
                model = TFIDF(n_gram_range=(3, 3), min_similarity=link_min_similarity)
            elif self.method in ["EditDistance", "Edit Distance"]:
                model = RapidFuzz()
            elif self.method in ["Embeddings", "Embedding"]:
                model = Embeddings(min_similarity=link_min_similarity)
            else:
                raise ValueError("Please instantiate the model with one of the following methods: \n"
                                 "* 'TF-IDF'\n"
                                 "* 'EditDistance'\n"
                                 "* 'Embeddings'\n"
                                 "* Or None if you want to automatically use TF-IDF")

        # Use TF-IDF if no model is specified
        elif not model:
            model = TFIDF(n_gram_range=(3, 3), min_similarity=link_min_similarity)

        # Group per model
        for name, match in self.matches.items():
            self._create_groups(name, model, link_min_similarity, group_all_strings)

    def get_ids(self) -> Union[str, List[str], None]:
        """ Get all model ids for easier access """
        check_matches(self)

        if isinstance(self.method, str):
            return self.method
        elif isinstance(self.method, Iterable):
            return [model.model_id for model in self.method]
        return None

    def get_matches(self, model_id: str = None) -> Union[pd.DataFrame,
                                                         Mapping[str, pd.DataFrame]]:
        """ Get the matches from one or more models"""
        check_matches(self)

        if len(self.matches) == 1:
            return list(self.matches.values())[0]

        elif len(self.matches) > 1 and model_id:
            return self.matches[model_id]

        return self.matches

    def get_clusters(self, model_id: str = None) -> Mapping[str, List[str]]:
        """ Get the groupings/clusters from a single model

        Arguments:
            model_id: the model id of the model if you have specified multiple models

        """
        check_matches(self)
        check_grouped(self)

        if len(self.matches) == 1:
            return list(self.clusters.values())[0]

        elif len(self.matches) > 1 and model_id:
            return self.clusters[model_id]

        return self.clusters

    def get_cluster_mappings(self, name: str = None) -> Mapping[str, int]:
        """ Get the mappings from the `To` column to its respective column """
        check_matches(self)
        check_grouped(self)

        if len(self.matches) == 1:
            return list(self.cluster_mappings.values())[0]

        elif len(self.matches) > 1 and name:
            return self.cluster_mappings[name]

        return self.cluster_mappings

    def save(self, path: str) -> None:
        """ Saves the model to the specified path

        Arguments:
            path: the location and name of the file you want to save
        
        Usage:
        ```python
        model.save("my_model")
        ```
        """
        with open(path, 'wb') as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, path: str):
        """ Loads the model from the specified path

        Arguments:
            path: the location and name of the PolyFuzz file you want to load
        
        Usage:
        ```python
        PolyFuzz.load("my_model")
        ```
        """
        with open(path, 'rb') as file:
            model = joblib.load(file)
        return model

    def _create_groups(self,
                       name: str,
                       model: BaseMatcher,
                       link_min_similarity: float,
                       group_all_strings: bool):
        """ Create groups based on either the To mappings if you compare two different lists of strings, or
        the From mappings if you compare lists of strings that are equal (set group_all_strings to True)
        """

        if group_all_strings:
            strings = list(self.matches[name].From.dropna().unique())
        else:
            strings = list(self.matches[name].To.dropna().unique())

        # Create clusters
        matches = model.match(strings)
        clusters, cluster_id_map, cluster_name_map = single_linkage(matches, link_min_similarity)

        # Map the `to` list to groups
        df = self.matches[name]
        df["Group"] = df['To'].map(cluster_name_map).fillna(df['To'])
        self.matches[name] = df

        # Track clusters and their ids
        self.clusters[name] = clusters
        self.cluster_mappings[name] = cluster_id_map

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
