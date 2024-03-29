import logging.config
import os.path
import yaml
from sklearn.model_selection import GridSearchCV
from src.constants import MODELS, script_folder, features
from src.features import FeatureRegistry
from src.model_factory import ModelFactory
from src.experiment import Experiment


def main():
    """
    Main function for the running experiments. You have to initialize a FeatureRegistry and pass it to the ModelFactory
    object as well as an Experiment object.

    Here we also load the logging config.
    """
    rg = FeatureRegistry()
    factory = ModelFactory(rg)
    exp = Experiment()

    rg.register_features(features, 'featureset1')

    with open(os.path.join(script_folder, '../logging_conf.yaml'), 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger(os.environ.get('MODE'))

    logger.debug('now running experiments for {}'.format(MODELS))
    for model in MODELS:
        mdl, search_space = factory.generate_model_and_search_space(mdl_type=model, feature_set_name='featureset1')
        with exp:
            exp.run_experiment(mdl)


if __name__ == "__main__":
    main()
