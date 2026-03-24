import re

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import gensim.downloader as api
from gensim.models import Word2Vec

def preprocess_text(text):
    
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lematize the words
    # TODO: estudar lematização e implementar um lematizador na mão
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Combine words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text


def train_svm_with_representations(train_data, test_data, representation):
    if representation == 'bow':
        vectorizer = CountVectorizer()
    elif representation == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid Representation. Choose 'bow' or 'tfidf'")
    
    X_train = vectorizer.fit_transform(train_data[0])
    y_train = train_data[1]
    X_test = vectorizer.transform(test_data)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


def get_word2vec_embeddings(data):
    tokenized_sentences = [sentence.split() for sentence in data]
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    embeddings = np.array([np.mean([model.wv[word] for word in sentence], axis=0) for sentence in tokenized_sentences])

    return embeddings


def train_svm_with_word2vec(train_data, test_data):
    X_train = get_word2vec_embeddings(train_data[0])
    y_train = train_data[1]
    X_test = get_word2vec_embeddings(test_data)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


def get_google_word2vec_embeddings(data):
    model = api.load('word2vec-google-news-300')

    tokenized_sentences = [sentence.split() for sentence in data]
    embeddings = []

    for sentence in tokenized_sentences:
        sentence_embeddings = []
        for word in sentence:
            if word in model:
                sentence_embeddings.append(model[word])
            
        if sentence_embeddings:
            embeddings.append(np.mean(sentence_embeddings, axis=0))

        else:
            sentence_embeddings.append(np.zeros(300))

    return np.array(embeddings)


def train_svm_with_google_word2vec(train_data, test_data):
    X_train = get_google_word2vec_embeddings(train_data[0])
    y_train = train_data[1]
    X_test = get_google_word2vec_embeddings(test_data)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


def main():

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    path_train = 'inteligencia_artificial/processamento_de_linguagem_natural/analise_de_sentimentos/python/data/Train.csv' 
    path_test = 'inteligencia_artificial/processamento_de_linguagem_natural/analise_de_sentimentos/python/data/Test.csv'
    path_valid = 'inteligencia_artificial/processamento_de_linguagem_natural/analise_de_sentimentos/python/data/Valid.csv'

    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    valid = pd.read_csv(path_valid)

    # Fiz testes com 3000 e a diferença em acurácia foi mínima. 
    # No trade-off com velocidade de treinamento, 2000 venceu.
    train = train.iloc[:2000]
    test = test.iloc[:2000]
    valid = test.iloc[:2000]

    train['preprocessed_text'] = train['text'].apply(preprocess_text)
    test['preprocessed_text'] = test['text'].apply(preprocess_text)
    valid['preprocessed_text'] = valid['text'].apply(preprocess_text)

    X_train = train['preprocessed_text']
    X_test = test['preprocessed_text']

    y_train = train['label']
    y_test = test['label']

    y_pred_bow = train_svm_with_representations([X_train, y_train], X_test, 'bow')
    accuracy_bow = accuracy_score(y_test, y_pred_bow)
    print(accuracy_bow)

    y_pred_tfidf = train_svm_with_representations([X_train, y_train], X_test, 'tfidf')
    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
    print(accuracy_tfidf)

    y_pred_word2vec = train_svm_with_word2vec([X_train, y_train], X_test)
    accuracy_word2vec = accuracy_score(y_test, y_pred_word2vec)
    print(accuracy_word2vec)

    y_pred_google_word2vec = train_svm_with_google_word2vec([X_train, y_train], X_test)
    accuracy_google_word2vec = accuracy_score(y_test, y_pred_google_word2vec)
    print(accuracy_google_word2vec)


if __name__ == '__main__':
    main()