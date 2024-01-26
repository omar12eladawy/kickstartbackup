import inspect
import os
import pickle
import time
from datetime import datetime

import logging

logger = logging.getLogger(os.environ.get('MODE'))
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import FEATURES, raw_data_path, data_file, TARGET, RANDOM_SEED
import sklearn
from sklearn import metrics
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import results_path


def _prepare_data():
    raw_data = os.path.join(raw_data_path, data_file)
    df = pd.read_csv(raw_data, index_col='Id')

    y = df[TARGET]
    X = df.drop(TARGET, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test


class Experiment(object):
    """
    A Class to help with running experiments. Also supports using with the "with" keyword. The actual functionalities are pretty basic since the core idea is to
    be replaced with the mlflow runner
    """

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.model = None
        self.results = None

    def __enter__(self):
        self.start_time = 0
        self.start_time = time.time()
        self.results = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is None:
            raise Exception("You need to run run_experiment(<model/hyperparametertuner>). Possible that there was an "
                            "error in the "
                            "fitting of the model")
        self.end_time = 0
        self.end_time = time.time()
        self.model = None
        logger.info("Elapsed Time %0.3f", self.end_time - self.start_time)
        logger.info("####### End of Experiment ####### \n")

    def run_experiment(self, experiment_object):
        """
        Runs the experiment based on the type of the object
        :param experiment_object: Object to be fitted. Can be a model or can be
        :return: None
        """
        now = datetime.now()
        timestamp = now.strftime("%d%m%y_%H%M")
        logger.info(
            ""
            "\n\n\t\t\t\t####### Starting Experiment dated {} #######\n\n".format(
                timestamp
            )
        )
        model, metrics_dict = self._get_model_and_results(experiment_object)
        pickle.dump(model, open(os.path.join(results_path, timestamp + type(model).__name__ + '.pkl'), 'wb'))

    def _get_model_and_results(self, experiment_object):
        """
        Getter method to help with fitting the different experiment objects.
        :param experiment_object: Object to be fitted. Can be a model or can be
        :return: None
        """
        ensembles = [x[1] for x in inspect.getmembers(sklearn.ensemble, inspect.isclass)]
        tuners = [x[1] for x in inspect.getmembers(sklearn.model_selection)]

        X_train, X_test, y_train, y_test = _prepare_data()

        if type(experiment_object) in ensembles:
            logger.info("Model type: %s", type(experiment_object).__name__)
            metrics_dict = {}

            logger.debug("The experiment object is an ensemble")
            experiment_object.fit(X_train, y_train)
            y_pred = experiment_object.predict(X_test)

            metrics_dict['accuracy'] = metrics.accuracy_score(y_test, y_pred) * 100
            metrics_dict['confusion matrix'] = confusion_matrix(y_test, y_pred)
            metrics_dict['classification report'] = classification_report(y_test, y_pred)

            logger.debug('The model is %s', self.model)
            logger.info(metrics_dict['classification report'])

            self.model = experiment_object
            return self.model, metrics_dict

        elif type(experiment_object) in tuners:
            logger.info("Hyperparameter tuning experiment")

            metrics_dict = {}
            logger.debug("The experiment object is a tuner ")
            experiment_object.fit(X_train, y_train)

            logger.info("Model type: %s", type(experiment_object.best_estimator_).__name__)
            self.model = experiment_object.best_estimator_

            y_pred = self.model.predict(X_test)

            metrics_dict['accuracy'] = metrics.accuracy_score(y_test, y_pred) * 100
            metrics_dict['confusion matrix'] = confusion_matrix(y_test, y_pred)
            metrics_dict['classification report'] = classification_report(y_test, y_pred)

            metrics_dict['best params'] = experiment_object.best_params_
            logger.debug('The model is %s', self.model)
            logger.info("Best parameter (CV score=%0.3f):", experiment_object.best_score_)
            logger.info("That following models had the following best parameters %s", experiment_object.best_params_)

            return self.model, metrics_dict
        else:
            raise NotImplementedError
