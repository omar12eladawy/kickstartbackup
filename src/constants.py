import logging
import os
import sys

from src.features import Feature, FeatureType
from src.models import Models
#TODO replace name with data from pd series right now this is very redundant
features = {
    'Elevation': Feature(name='Elevation', feat_type=FeatureType.NUM),
    'Aspect': Feature(name='Aspect', feat_type=FeatureType.NUM),
    'Slope': Feature(name='Slope', feat_type=FeatureType.NUM),
    'Horizontal_Distance_To_Hydrology': Feature(name='Horizontal_Distance_To_Hydrology', feat_type=FeatureType.NUM),
    'Vertical_Distance_To_Hydrology': Feature(name='Vertical_Distance_To_Hydrology', feat_type=FeatureType.NUM),
    'Horizontal_Distance_To_Roadways': Feature(name='Horizontal_Distance_To_Roadways', feat_type=FeatureType.NUM),
    'Hillshade_9am': Feature(name='Hillshade_9am', feat_type=FeatureType.NUM),
    'Hillshade_Noon': Feature(name='Hillshade_Noon', feat_type=FeatureType.NUM),
    'Hillshade_3pm': Feature(name='Hillshade_3pm', feat_type=FeatureType.NUM),
    'Horizontal_Distance_To_Fire_Points': Feature(name='Horizontal_Distance_To_Fire_Points', feat_type=FeatureType.NUM),
    'Wilderness_Area': Feature(name='Wilderness_Area', feat_type=FeatureType.CAT),
    'Soil_Type': Feature(name='Soil_TypeCAT', feat_type=FeatureType.CAT),

}
# Experiment Constants
FEATURES = {
    'cat_cols': ['Wilderness_Area', 'Soil_Type'],
    'num_cols': [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points']
}
TARGET = 'Cover_Type'
RANDOM_SEED = 42
MODELS = [Models.RANDOM_FOREST]

# Path Constants
data_file = 'forest_data.csv'

src_folder = os.path.dirname(os.path.abspath("__file__"))
script_folder = os.path.join(src_folder, "../scripts")
destination_path = os.path.join(src_folder, "../data")
raw_data_path = os.path.join(src_folder, "../data/raw")
results_path = os.path.join(src_folder, "../results")
