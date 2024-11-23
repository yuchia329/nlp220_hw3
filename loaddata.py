import numpy as np
import torch
import random
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import json
from torch.utils.data import DataLoader, TensorDataset
import nltk
import re
import string


def downloadNLTK():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(articles)]

def preprocess_text(abstract):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()

    """
    Preprocess a single scientific abstract.
    """
    # Step 1: Lowercase the text
    abstract = abstract.lower()

    # Step 2: Remove citations (e.g., "[1]", "(Smith et al., 2020)")
    abstract = re.sub(r'\[\d+\]', '', abstract)  # Remove [1], [2]
    abstract = re.sub(r'\([a-zA-Z]+ et al\., \d{4}\)', '', abstract)  # Remove (Smith et al., 2020)

    # Step 3: Remove special characters, punctuation, and numbers
    abstract = re.sub(r'\d+', '', abstract)  # Remove digits
    abstract = abstract.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

    # Step 4: Tokenize the text
    tokens = nltk.tokenize.word_tokenize(abstract)

    # Step 5: Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Step 6: Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 7: Remove short tokens (e.g., "a", "x") to focus on meaningful words
    tokens = [word for word in tokens if len(word) > 2]

    return " ".join(tokens)

def preProcessText(text, isalpha=False, stopwords=False):
    # downloadNLTK()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()] if isalpha else tokens  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words] if stopwords else tokens  # Remove stopwords
    return " ".join(tokens)

def loadJSON(file_path):
    df = pd.read_json(file_path)
    return df

def splitXY(df):
    # x = pd.DataFrame({"text": df["summaries"].values})["text"].values
    x = pd.DataFrame({"text": df["titles"].values + " " + df["summaries"].values})["text"].values
    y = pd.get_dummies(df['terms'].explode()).groupby(level=0).sum()
    with open("token_freq_acc.json", "r") as f:
        acc_data = json.load(f)
        keys = list(acc_data.keys())
        y = y[keys]
    return x, y

def splitDataset(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1234)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1234)
    return x_train, y_train, x_val, y_val, x_test, y_test

def trimLabels(y, reduce_labels_rate):
    with open("token_freq_acc.json", "r") as f:
        acc_data = json.load(f)
        retain_index = 0
        for index, key in enumerate(acc_data.keys()):
            if acc_data[key] > reduce_labels_rate:
                retain_index = index
                break
        y = y[list(acc_data.keys())[:retain_index+1]]
        return y

def seed_worker(worker_id):
    np.random.seed(42)
    random.seed(42)

def makeDateloader(XY_sets):
    dataloaders= []
    datasets = []
    batch_size=32
    for (x_vec, y_vec, shuffle) in XY_sets:
        dataset = TensorDataset(torch.FloatTensor(x_vec), torch.FloatTensor(y_vec))
        dataloader = DataLoader(dataset, batch_size, pin_memory=True, shuffle=shuffle, worker_init_fn=seed_worker)
        dataloaders.append(dataloader)
        datasets.append(dataset)
    return dataloaders, datasets

def prepareData(file_path, reduce_labels_rate=0):
    df = loadJSON(file_path)
    x, y = splitXY(df)
    downloadNLTK()
    x = [preprocess_text(doc) for doc in x]
    x_train, y_train, x_val, y_val, x_test, y_test = splitDataset(x,y)
    if reduce_labels_rate > 0:
        y_train = trimLabels(y_train, reduce_labels_rate)
    
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=99999)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)
    y_train_vec = y_train.to_numpy()
    y_val_vec = y_val.to_numpy()
    y_test_vec = y_test.to_numpy()

    # using scipy.sparse.hstack for efficient sparse matrix operations
    x_train_vec = hstack([x_train_vec])
    x_val_vec = hstack([x_val_vec])
    x_test_vec = hstack([x_test_vec])
    
    # Scale features
    scaler = StandardScaler(with_mean=False)
    x_train_vec = scaler.fit_transform(x_train_vec)
    x_val_vec = scaler.transform(x_val_vec)
    x_test_vec = scaler.transform(x_test_vec)
    return x_train_vec, x_val_vec, x_test_vec, y_train_vec, y_val_vec, y_test_vec
