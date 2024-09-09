import pandas as pd
import numpy as np

import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")






# importing the data

# df = pd.read_csv('sentiment_data_label.csv')
# print(df.head())

# print(df.columns)

# df.drop(columns=['Unnamed: 0', 'Date', 'class'], axis=1, inplace=True)

# print(df.head())


data = {"text": ["hello ###this is a link https://chatgpt.com, i'm @hostel",
                "#hello this singing is google link http://google.com@whatsapp @insta",
                "hello this is mylink improving  https://github",
                "hellooooooooooooo this is random swimming eating link https://random",
                "The swimming pool is beatuiful"]}


def remove_links(df, column):
    for i in range(len(df)):
        df.loc[i, column] = re.sub(r"http\S+", "", df.loc[i, column])

def remove_mentions(df, column):
    for i in range(len(df)):
        df.loc[i, column] = re.sub(r"@", "", df.loc[i, column])

def remove_hashtags(df, column):
    for i in range(len(df)):
        df.loc[i, column] = re.sub(r"#", "", df.loc[i, column])

def remove_special_chars(df, column):
    for i in range(len(df)):
        df.loc[i, column] = re.sub(r"[^a-zA-Z0-9\s]", "", df.loc[i, column])


def to_lowercase(df, column):
    for i in range(len(df)):
        df.loc[i, column] = df.loc[i, column].lower()


def tokenize_text(df, column):
    data = []
    for i in range(len(df)):
        data.append(word_tokenize(df.loc[i, column]))
    return data


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    data = []
    for i in range(len(tokens)):
        data.append([word for word in tokens[i] if word not in stop_words])
    return data


def lemmatize_words_spacy(tokens):
    lemmatized_data = []
    for sentence in tokens:
        doc = nlp(" ".join(sentence))
        lemmatized_sentence = [token.lemma_ for token in doc]
        lemmatized_data.append(lemmatized_sentence)
    return lemmatized_data


def combine(tokens):
    data = []
    for i in range(len(tokens)):
        data.append(" ".join(tokens[i]))
    return data



def preprocess_data(df, column):
    remove_links(df, column)
    remove_mentions(df, column)
    remove_hashtags(df, column)
    remove_special_chars(df, column)
    to_lowercase(df, column)
    words = tokenize_text(df, column)
    words = remove_stopwords(words)
    words = lemmatize_words_spacy(words)
    return words




df = pd.DataFrame(data)
print(df['text'])
print("\n\n") 
print("data after cleaning\n\n")
df['final text'] = preprocess_data(df, 'text') ;
print(df['final text'])


word2vec_model = Word2Vec(df['final text'], vector_size=100, window=5, min_count=1, workers=4)


def get_sentence_vector(words, model):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    sentence_vector = np.mean(word_vectors, axis=0)  
    return sentence_vector

# # Apply to all sentences
sentence_vectors = np.array([get_sentence_vector(sentence, word2vec_model) for sentence in df['final text']])
print(sentence_vectors)


# now we are ready to build the model