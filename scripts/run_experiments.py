import logging.config
import os.path
import yaml
from sklearn.model_selection import GridSearchCV
from src.constants import MODELS, script_folder
from src.features import FeatureRegistry
from src.model_factory import ModelFactory
from src.experiment import Experiment


def main(models):
    logging.debug('The models passed are: {}', models)

    rg = FeatureRegistry()
    factory = ModelFactory(rg)
    exp = Experiment()

    with open(os.path.join(script_folder, '../logging_conf.yaml'), 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger(os.environ.get('MODE'))

    logger.info('now running experiments for {}'.format(MODELS))
    for model in MODELS:
        mdl, search_space = factory.generate_model_and_search_space(mdl_type=model)
        with exp:
            search = GridSearchCV(mdl, search_space, n_jobs=2)
            exp.run_experiment(search)


if __name__ == "__main__":
    main()
