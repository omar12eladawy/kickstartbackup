import itertools
from enum import Enum
from typing import Dict


class FeatureType(Enum):
    CAT = 'categorical',
    NUM = 'numerical'


class Feature:
    def __init__(self, name: str, feat_type: FeatureType):
        self.name = name
        self.feat_type = feat_type


def _get_categorical(features):
    return_dict = {}
    for k, v in features.items():
        if v.feat_type is FeatureType.CAT:
            return_dict[k] = v
    return return_dict


def _get_numerical(features):
    return_dict = {}
    for k, v in features.items():
        if v.feat_type is FeatureType.NUM:
            return_dict[k] = v
    return return_dict


class FeatureRegistry:
    def __init__(self):
        self.registry = {}

    def register_features(self, features: Dict, feature_set_name: str):
        cat_features = _get_categorical(features)
        num_features = _get_numerical(features)

        fts = {
            "cat_features": cat_features,
            "num_features": num_features
        }

        self.registry[feature_set_name] = fts

    def get_features(self, feature_set_name, feature_type: FeatureType = None):
        if bool(self.registry) is False:
            raise Exception(
                "There are no feature_sets registered. Make sure to register the features with FeatureRegistry."
                "register_features first")

        fts = self.registry[feature_set_name]
        if feature_type is FeatureType.CAT:
            return fts["cat_features"]
        elif feature_type is FeatureType.NUM:
            return fts["num_features"]
        else:
            td = list(itertools.chain(fts["cat_features"], fts["num_features"]))
            return list(itertools.chain(fts["cat_features"], fts["num_features"]))

