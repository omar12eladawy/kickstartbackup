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
    feature_list = []
    for k, v in features.items():
        if v.feat_type is FeatureType.CAT:
            feature_list.append({k: v})
    return feature_list


def _get_numerical(features):
    feature_list = []
    for k, v in features.items():
        if v.feat_type is FeatureType.NUM:
            feature_list.append({k: v})
    return feature_list


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
        return_list = []
        if feature_type is FeatureType.CAT:
            return_list.append(fts["cat_features"])
        elif feature_type is FeatureType.NUM:
            return_list.append(fts["num_features"])
        else:
            return_list.append(fts["cat_features"])
            return_list.append(fts["num_features"])
        return return_list


if __name__ == "__main__":
    fts = {
        'Elevation': Feature(name='Elevation', feat_type=FeatureType.NUM),
        'Aspect': Feature(name='Aspect', feat_type=FeatureType.NUM),
        'Slope': Feature(name='Slope', feat_type=FeatureType.NUM),
        'Horizontal_Distance_To_Hydrology': Feature(name='Horizontal_Distance_To_Hydrology', feat_type=FeatureType.NUM),
        'Vertical_Distance_To_Hydrology': Feature(name='Vertical_Distance_To_Hydrology', feat_type=FeatureType.NUM),
        'Horizontal_Distance_To_Roadways': Feature(name='Horizontal_Distance_To_Roadways', feat_type=FeatureType.NUM),
        'Hillshade_9am': Feature(name='Hillshade_9am', feat_type=FeatureType.NUM),
        'Hillshade_Noon': Feature(name='Hillshade_Noon', feat_type=FeatureType.NUM),
        'Hillshade_3pm': Feature(name='Hillshade_3pm', feat_type=FeatureType.NUM),
        'Horizontal_Distance_To_Fire_Points': Feature(name='Horizontal_Distance_To_Fire_Points',
                                                      feat_type=FeatureType.NUM),
        'Wilderness_Area': Feature(name='Wilderness_Area', feat_type=FeatureType.CAT),
        'Soil_Type': Feature(name='Soil_TypeCAT', feat_type=FeatureType.CAT),

    }
    rg = FeatureRegistry()
    rg.register_features(fts, 'test_featureest')
