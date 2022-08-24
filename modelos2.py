from enum import Enum
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class Models2(Enum):
    BAYES = 'Bayes'
    SVM_LINEAR = 'SVM_Linear'
    MLP_1LAYER = 'MLP_1Layer'

class Vectorizers2(Enum):
    TFIDF = 'TFIDF'
    BOW = 'BOW'

VECTORIZERS2 = {
    Vectorizers2.TFIDF.value: TfidfVectorizer(),
    Vectorizers2.BOW.value: CountVectorizer(),
}

CLF_MODELS2 = {
    Models2.BAYES.value: MultinomialNB(),
    Models2.SVM_LINEAR.value: SVC(kernel='linear'),
    Models2.MLP_1LAYER.value: MLPClassifier(early_stopping=True),
}


C = [2 ** i for i in range(-5, 20, 2)]

CLF_PARAMS2 = {
    Models2.BAYES.value: {
        'var_smoothing': np.logspace(0,-9, num=50)
    },
    Models2.SVM_LINEAR.value: {
        'C': C
    },
    Models2.MLP_1LAYER.value: {
        'hidden_layer_sizes': list(np.arange(5, 300, 35))
    },
}

VECTORIZERS_PARAMS2 = {
    'max_df': (0.2, 0.5, 0.9)
    }