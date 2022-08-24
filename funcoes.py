import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

# Remoção de stop words:
class Stopword():

    def re_stopword(self, text):
        stopwords =  nltk.corpus.stopwords.words('english')
        words = [ word for word in word_tokenize(text) if not word in stopwords]
        new_text =  " ".join(words)
        return new_text

# Contagem de palavras:
class Count():
    def contagem(self, df):
        df = df.assign(number_words = df['limpeza_simples'].apply(lambda x: len(x.split(" ")))) 
        df = df.drop(df[df['number_words'] < 3].index) 
        return df

# Remover link:
class Link():
    def re_links_and_users(self, text):
            words = text.split()
            words = [word for word in words if ("@" not in word) and ("http" not in word)]
            new_text =  " ".join(words)
            return new_text

# Limpeza manual:
class Sub_re():
    def sub_re(self, text):
        new_text = re.sub(r'wooooooooooooooooooooooooooow','wow',text)
        new_text = re.sub(r'sexsex','sex',new_text)
        new_text = re.sub(r'chris paul ','',new_text)
        new_text = re.sub(r'steve','',new_text)
        new_text = re.sub(r'chinese','',new_text)
        new_text = re.sub(r'jordan','',new_text)
        new_text = re.sub(r'mwahahahahahahahahahahahahahahahahahahahaha','',new_text)
        new_text = re.sub(r'gigowngnognronwoigwnoirowinowrioirwnorwoinrwoingroinrwoingroinoingwoingoinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnw','',new_text)
        new_text = re.sub(r'ewwww','',new_text)
        return new_text

# Removendo com um caracteres:
class One_carct():
    def words_one(self, text):
        words = [ word for word in word_tokenize(text) if len(word) > 1]
        new_text =  " ".join(words)
        return new_text

# Lematização
class Lematization():
    def lemmatization(self, text):
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        new_text =  " ".join(lemmas)
        return new_text

# Stemming
class Stemming():
    def stemming(self, text):
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word) for word in text.split()]
        new_text =  " ".join(words)
        return new_text