from enum import Enum


class Models(str, Enum):
    """
    Enum class to help with typing and auto-completion in the IDE
    """
    GRADIENT_BOOSTING = 'Gradient Boosting',
    GAUSSIAN_NB = 'Gaussian Naive Bayes',
    LOG_REGRESSION = 'Logistic Regression',
    MLP_CLASSIFIER = 'MLP Classifier',
    KNN = 'KNN',
    RANDOM_FOREST = 'Random Forest',
    DECISION_TREE = 'Decision Tree',
