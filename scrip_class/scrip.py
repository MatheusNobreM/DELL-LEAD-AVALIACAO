from sklearn.feature_extraction.text import CountVectorizer
import joblib
import importlib
import funcoes
importlib.reload(funcoes)

def limpeza(text):
    link = funcoes.Link()
    limpo = link.links(text)
    min = funcoes.Minusculo()
    limpo = min.minusculo(limpo)
    ls = funcoes.Limpeza_Simples()
    limpo = ls.limpeza_simples(limpo)
    regex = funcoes.Sub_re()
    limpo = regex.sub_re(limpo)
    one = funcoes.One_carct()
    limpo = one.words_one(limpo)
    stem = funcoes.Stemming()
    limpo = stem.stemming(limpo)
    return limpo


def Predicted(text):
    new_text = limpeza(text)
    clf = joblib.load('/home/matheus/Documents/Mentoria/Avaliação Final/Melhor_modelo/best_modelo.pkl')
    text_predicted = clf.predict([new_text])
    new_predicted = text_predicted[0]
    if new_predicted == 0:
        type_clf = 'Non-Toxic'
    else:
        type_clf = 'Toxic'

    return type_clf