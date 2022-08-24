from enum import Enum
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Models(Enum):
    BAYES = 'Bayes'

class Vectorizers(Enum):
    BOW = 'BOW'

VECTORIZERS = {
    Vectorizers.BOW.value: CountVectorizer(),
}

CLF_MODELS = {
    Models.BAYES.value: MultinomialNB(),
}

CLF_PARAMS = {
    Models.BAYES.value: {
        'var_smoothing': np.logspace(0,-9, num=30)}
}

VECTORIZERS_PARAMS = {
    'max_df': ( 0.5, 0.8, 0.9, 1),
    }