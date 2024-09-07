import pandas as pd
import numpy as np

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# importing the data

# df = pd.read_csv('sentiment_data_label.csv')
# print(df.head())

# print(df.columns)

# df.drop(columns=['Unnamed: 0', 'Date', 'class'], axis=1, inplace=True)

# print(df.head())


data = {"text": ["hello ###this is a link https://chatgpt.com, i'm @hostel",
                "#hello this is google link http://google.com@whatsapp @insta",
                "hello this is mylink https://github",
                "hello this is random link https://random"]}


def remove_links(df):
    link_free = []
    for i in range(df.shape[0]):
        link_free.append(re.sub(r"http\S+", "", df.iloc[i]['text']))
    return link_free

def remove_mentions(df):
    data = []
    for i in range(df.shape[0]):
        data.append(re.sub(r"@w+", "", df.iloc[i]['refine text']))
    return data

def remove_hashtags(df):
    data = []
    for i in range(df.shape[0]):
        data.append(re.sub(r"#", "", df.iloc[i]['removed mentions']))
    return data


def remove_special_chars(text):
    data = []
    for i in range(df.shape[0]):
        data.append(re.sub(r"[^a-zA-Z0-9\s]", "", df.iloc[i]['removed mentions']))
    return data


def to_lowercase(text):
    data = []
    for i in range(df.shape[0]):
        data.append(df.iloc[i]['removed mentions'].lower())
    return data

def tokenize_text(df):
    data = []
    for i in range(df.shape[0]):
        data.append(word_tokenize(df.iloc[i]['removed hash'].lower()))
    return data


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    data = []
    for i in range(len(tokens)):
        data.append([word for word in tokens[i] if word not in stop_words])
    return data


def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    data = []
    for i in range(len(tokens)):
        data.append([lemmatizer.lemmatize(word) for word in tokens[i]])
    return data


def combine(tokens):
    data = []
    for i in range(len(tokens)):
        data.append(" ".join(tokens[i]))
    return data 

df = pd.DataFrame(data)
print(df)
print("data after cleaning\n\n")
df['refine text'] = remove_links(df)
df['removed mentions'] = remove_mentions(df) 
df['removed hash'] = remove_hashtags(df)
# print(df['removed hash'])
words = tokenize_text(df)
updated_words = remove_stopwords(words)
roots_words = lemmatize_words(updated_words)
df['final text'] = combine(roots_words)
print(df['final text'])
# words(tokens)  # Or stem_words(tokens) for stemming
# cleaned_text = " ".j