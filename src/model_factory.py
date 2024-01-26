import os
from typing import Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier

from src.features import FeatureRegistry
from src.models import Models
import logging

logger = logging.getLogger(os.environ.get('MODE'))

class ModelFactory:
    """
    Factory class the implements methods for the creation of models and their search spaces. The Factory is by
    design stateless (not attributes that needs to be initialized)
    """
    def __init__(self, rg: FeatureRegistry):
        self.rg = rg
    def generate_model_and_search_space(self, mdl_type: Models) -> Tuple:
        """
        Method that return the chosen model instance as well as a search space for optimization
        :param mdl_type: Type of model. If not implemented raise error.
        :return: Tuple of model
        """
        model = self._get_model(mdl_type)
        search_space = self._get_search_space(mdl_type)
        return model, search_space

    def _get_model(self, mdl_type: Models):
        """
        Method to fetch the correct model type
        :param mdl_type: the type to model based on the Enum model class
        :return: an Instance of the model
        """
        if mdl_type == Models.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(n_estimators=50,
                                              learning_rate=0.09,
                                              max_depth=5,
                                              verbose=True)
        elif mdl_type == Models.GAUSSIAN_NB:
            return GaussianNB()
        elif mdl_type == Models.MLP_CLASSIFIER:
            return MLPClassifier(solver='adam',
                                 alpha=1e-5,
                                 hidden_layer_sizes=(5, 2),
                                 random_state=1,
                                 max_iter=1000,
                                 verbose=True)
        elif mdl_type == Models.KNN:
            return KNeighborsClassifier(n_neighbors=5)
        elif mdl_type == Models.RANDOM_FOREST:
            return RandomForestClassifier(n_estimators=100)
        elif mdl_type == Models.DECISION_TREE:
            return DecisionTreeClassifier(random_state=42, max_depth=2)
        elif mdl_type == Models.LOG_REGRESSION:
            return linear_model.LogisticRegression()
        # solver = 'liblinear',C=10, penalty='l2', max_iter = 1000
        else:
            raise NotImplementedError

    def _get_search_space(self, mdl_type: Models):
        """logger
        A method to get the hyperparameter search space
        :param mdl_type:  mdl_type: the type to model based on the Enum model class
        :return:  The Hyperparameter search space
        """
        space = dict()
        if mdl_type == Models.GRADIENT_BOOSTING:
            logger.warning("The search space for %s has not been implemented yet", mdl_type.value[0])
            return None
        elif mdl_type == Models.GAUSSIAN_NB:
            logger.warning("The search space for %s has not been implemented yet", mdl_type.value[0])
            return None
        elif mdl_type == Models.MLP_CLASSIFIER:
            logger.warning("The search space for %s has not been implemented yet", mdl_type.value[0])
            return None
        elif mdl_type == Models.KNN:
            logger.warning("The search space for %s has not been implemented yet", mdl_type.value[0])
            return None
        elif mdl_type == Models.RANDOM_FOREST:
            space['bootstrap'] = [False, True]
            space['max_features'] = ['log2', 'sqrt']
            space['n_estimators'] = [50, 100, 150, 200]
            return space
        elif mdl_type == Models.DECISION_TREE:
            logger.warn("The search space for the {} has not been implemented yet", mdl_type.value[0])
            return None
        elif mdl_type == Models.LOG_REGRESSION:
            space['solver'] = ['newton-cg', 'liblinear']
            space['penalty'] = ['l1', 'l2']
            space['C'] = [1, 10, 20, 50]
            return space
        else:
            raise NotImplementedError
