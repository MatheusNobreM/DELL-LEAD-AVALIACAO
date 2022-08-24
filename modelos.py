from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


class Models(Enum):
    BAYES = 'Bayes'
    KNN = 'KNN'
    SVM_LINEAR = 'SVM_Linear'
    SVM_RBF = 'SVM_RBF'
    MLP_2LAYERS = 'MLP_2Layers'
    RANDOM_FOREST = 'Random_Forest'
    Logistic_Regression = 'Logistic_Regression'

class Vectorizers(Enum):
    TFIDF = 'TFIDF'
    BOW = 'BOW'

VECTORIZERS = {
    Vectorizers.TFIDF.value: TfidfVectorizer(),
    Vectorizers.BOW.value: CountVectorizer(),
}

CLF_MODELS = {
    Models.BAYES.value: MultinomialNB(),
    Models.KNN.value: KNeighborsClassifier(),
    Models.SVM_LINEAR.value: SVC(kernel='linear'),
    Models.SVM_RBF.value: SVC(kernel='rbf'),
    Models.MLP_2LAYERS.value: MLPClassifier(early_stopping=True),
    Models.RANDOM_FOREST.value: RandomForestClassifier(),
    Models.Logistic_Regression.value: LogisticRegression(),
}


C = [2 ** i for i in range(-10, 5, 2)]
gamma = [2 ** i for i in range(-10, 4, 2)]

CLF_PARAMS = {
    Models.BAYES.value: {
        'var_smoothing': np.logspace(0,-9, num=30)
    },
    Models.KNN.value: {
        'n_neighbors': [3, 7, 9, 15]
    },
    Models.SVM_LINEAR.value: {
        'C': C
    },
    Models.SVM_RBF.value: {
        'C': C,
        'gamma': gamma
    },
    Models.MLP_2LAYERS.value: {
        'hidden_layer_sizes': [(random.randrange(2, 500, 25), random.randrange(2, 500, 25)) for i in range(6)]
    },
    Models.RANDOM_FOREST.value:  {
        'n_estimators': list(np.arange(25, 50, 5)),
    },
    Models.Logistic_Regression.value: {
        'solver': ['lbfgs', 'sag', 'saga'],
    },
}

VECTORIZERS_PARAMS = {
    'max_df': (0.2, 0.5, 0.8, 0.9, 1),
    'ngram_range': [(1, 1), (1, 2), (1, 3)]
    }