from typing import List, Mapping, Union, Iterable
from polyfuzz.models import TFIDF, EditDistance, Embeddings
from polyfuzz.models.base import BaseMatcher
from polyfuzz.metrics import precision_recall_curve, visualize_precision_recall
from polyfuzz.models.utils import cluster_mappings


class PolyFuzz:

    def __init__(self,
                 method: Union[str, Mapping[str, BaseMatcher]],
                 from_list: List[str],
                 to_list: List[str]):
        self.method = method
        self.matches = None
        self.min_precisions = None
        self.recalls = None
        self.average_precisions = None
        self.clusters = None
        self.cluster_mappings = None
        self.grouped_matches = None
        self.from_list = from_list
        self.to_list = to_list

    def _update_model_ids(self):
        # Give models a model_id if it didn't already exist
        for index, model in enumerate(self.method):
            if not model.model_id:
                model.model_id = f"Model {index}"

        # Update duplicate names
        model_ids = [model.model_id for model in self.method]
        if len(set(model_ids)) != len(model_ids):
            for index, model in enumerate(self.method):
                model.model_id = f"Model {index}"

    def match(self):
        # Standard models - quick access
        if isinstance(self.method, str):
            if self.method == "TF-IDF":
                self.matches = {"TF-IDF": TFIDF(min_similarity=0).match(self.from_list, self.to_list)}
            elif self.method == "EditDistance":
                self.matches = {"EditDistance": EditDistance().match(self.from_list, self.to_list)}
            elif self.method == "Embeddings":
                self.matches = {"Embeddings": Embeddings(min_similarity=0).match(self.from_list, self.to_list)}

        # Custom models
        elif isinstance(self.method, Iterable):
            self._update_model_ids()
            self.matches = {model.model_id: model.match(self.from_list, self.to_list) for model in self.method}

        return self

    def visualize_precision_recall(self):
        self.min_precisions = {}
        self.recalls = {}
        self.average_precisions = {}

        for name, match in self.matches.items():
            min_precision, recall, average_precision = precision_recall_curve(match)
            self.min_precisions[name] = min_precision
            self.recalls[name] = recall
            self.average_precisions[name] = average_precision

        visualize_precision_recall(self.matches, self.min_precisions, self.recalls)

    def group(self, minimum_similarity=0.8):
        self.clusters = {}
        self.cluster_mappings = {}

        for name, match in self.matches.items():
            strings = list(self.matches[name].To.dropna().unique())
            tfidf, _ = TFIDF(n_gram_range=(3, 3))._extract_tf_idf(strings, None)
            clusters, cluster_id_map, cluster_name_map = cluster_mappings(tfidf, strings, minimum_similarity)
            self._map_groups(name, cluster_name_map)
            self.clusters[name] = clusters
            self.cluster_mappings[name] = cluster_id_map

    def _map_groups(self, name, mapping_dict):
        """ Map the 'to' list to groups """
        df = self.matches[name]
        df["Group"] = df['To'].map(mapping_dict).fillna(df['To'])

        # Fix that some mappings from "From" end up in "Group"
        df.loc[(df.From != df.To) &
               (df.From == df.Group), "Group"] = df.loc[(df.From != df.To) &
                                                        (df.From == df.Group), "To"]
        self.matches[name] = df

    def get_all_model_ids(self):
        if isinstance(self.method, str):
            return self.method
        elif isinstance(self.method, Iterable):
            return [model.model_id for model in self.method]
        return None

    def get_match(self, name):
        return self.matches[name]

    def get_clusters(self, name):
        return self.clusters[name]

    def get_cluster_mappings(self, name):
        return self.cluster_mappings[name]
