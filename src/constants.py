import logging
import os
import sys

from src.models import Models

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
