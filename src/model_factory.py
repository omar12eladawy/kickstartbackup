import os
from typing import Tuple

from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FeatureRegistry, FeatureType
from src.models import Models
import logging

logger = logging.getLogger(os.environ.get('MODE'))


# todo generate keys dynamically
class ModelFactory:
    """
    Factory class the implements methods for the creation of models and their search spaces. The Factory is by
    design stateless (not attributes that needs to be initialized)
    """

    def __init__(self, rg: FeatureRegistry):
        self.rg = rg

    def generate_model_and_search_space(self, mdl_type: Models, feature_set_name: str, feature_options=None) -> Tuple:
        """
        Method that return the chosen model instance as well as a search space for optimization
        :param mdl_type: Type of model. If not implemented raise error.
        :param feature_options:
        :return: Tuple of model
        """
        model = self._get_model(mdl_type, feature_set_name, feature_options)
        search_space = self._get_search_space(mdl_type)
        return model, search_space

    def _get_model(self, mdl_type: Models, feature_set_name, feature_options):

        """
        Method to fetch the correct model type
        :param mdl_type: the type to model based on the Enum model class
        :return: an Instance of the model
        """

        def _get_features(feature_set_name: str, feature_type: FeatureType = None):
            fts = self.rg.get_features(feature_set_name, feature_type)
            keys_list = [v.name for k, v in fts.items()]
            return keys_list

        if mdl_type == Models.GRADIENT_BOOSTING:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())
            pipe = Pipeline([
                ('transformer', col_transformer),
                ('gbc', GradientBoostingClassifier(n_estimators=50,
                                                   learning_rate=0.09,
                                                   max_depth=5,
                                                   verbose=True))])
            return pipe
        elif mdl_type == Models.GAUSSIAN_NB:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())
            pipe = Pipeline([
                ('transformer', col_transformer),
                ('gnb', GaussianNB())])
            return pipe
        elif mdl_type == Models.MLP_CLASSIFIER:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())
            pipe = Pipeline([
                ('transformer', col_transformer),
                ('mlpc', MLPClassifier(solver='adam',
                                       alpha=1e-5,
                                       hidden_layer_sizes=(5, 2),
                                       random_state=1,
                                       max_iter=1000,
                                       verbose=True))])
            return pipe
        elif mdl_type == Models.KNN:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())
            pipe = Pipeline([
                ('transformer', col_transformer),
                ('knn', KNeighborsClassifier(n_neighbors=5))])
            return pipe
        elif mdl_type == Models.RANDOM_FOREST:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())

            pipe = Pipeline([
                ('transformer', col_transformer),
                ('rf', RandomForestClassifier(n_estimators=100))])

            return pipe
        elif mdl_type == Models.DECISION_TREE:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())

            pipe = Pipeline([
                ('transformer', col_transformer),
                ('dt', DecisionTreeClassifier(random_state=42, max_depth=2))])

            return pipe
        elif mdl_type == Models.LOG_REGRESSION:
            cat_fts = _get_features(feature_set_name, feature_type=FeatureType.CAT)
            col_transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), cat_fts),
                remainder=StandardScaler())

            pipe = Pipeline([
                ('transformer', col_transformer),
                ('lr', linear_model.LogisticRegression())])

            return pipe
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
            space['rf__bootstrap'] = [False, True]
            space['rf__max_features'] = ['log2', 'sqrt']
            space['rf__n_estimators'] = [50, 100, 150, 200]
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
