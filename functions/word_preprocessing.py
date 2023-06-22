from typing import List
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from functions.utils import find_code_name_in_dict

import nltk
import re
import gensim.downloader as api
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()
vectorizer = api.load("word2vec-ruscorpora-300")


def preprocessing(sent: str):
    """Return negative and positive words.
    If there is the word 'Кроме', then the next words
    will be threated as negative

    Args:
        sent (str): sentence

    Returns:
        str: negative words
        str: positive words
    """
    minus = ""
    sent = sent.lower()
    sent = sent.replace(",", "")
    sent = sent.replace(";", "")
    if "кроме" in sent:
        if sent[sent.find("кроме")-1] == "(":
            minus = sent[sent.find("кроме")+5:sent.find(")")]
            sent = sent[:sent.find("кроме")-1] + sent[sent.find(")")+2:]
        else:
            minus = sent[sent.find("кроме")+5:]
            sent = sent[:sent.find("кроме")-1]
        sent = sent.replace("(", "")
        sent = sent.replace(")", "")
    return minus, sent

def lemmatize(doc) -> List[str]:
    """Remove stop words, transform to normal form
    and return the TAG of word

    Args:
        doc (str): sentence

    Returns:
        set: set of tokens
    """
    doc = re.sub(patterns, ' ', doc)
    tokens = [morph.normal_forms(token.strip())[0] for token in doc.split() if token not in stopwords_ru]
    tokens = [tkn + "_" + morph.parse(tkn)[0].tag.POS if morph.parse(tkn)[0].tag.POS != None else tkn + "_NONE" for tkn in tokens]
    return list(set(tokens))

def get_vector(pos_tokens, neg_tokens):
    """Return the sum for positive vectors,
    then subtract negative vector

    Args:
        rows (List[str]): Positive tokens
        minuses (List[str]): Negative tokens

    Returns:
        np.ndarray: vector
    """
    pos = np.sum([vectorizer[a] for a in pos_tokens if a in vectorizer.key_to_index], axis=0)
    if neg_tokens:
        neg = np.sum([vectorizer[a] for a in neg_tokens if a in vectorizer.key_to_index], axis=0)
        return (pos - neg).astype(np.float32)
    if len(pos.shape) == 0:
        return np.array([0] * 300).astype(np.float32)
    return pos.astype(np.float32)

def vectorize(sentence: str) -> np.ndarray:
    """
    Take sentence and perform:
    1. Preprocessing
    2. Lemmatize
    3. Get vector from 

    Args:
        sentence (str): sentence that we need to proceeed

    Returns:
        np.ndarray: word vector
    """
    negative_string, positive_string = preprocessing(sentence)
    positive_tokens, negative_tokens = lemmatize(positive_string), lemmatize(negative_string)
    vector = get_vector(positive_tokens, negative_tokens)
    # If our vector is the number -> return zeros like vector
    if isinstance(vector, np.float32):
        return np.zeros(shape=(300,), dtype=np.float32)
    return vector

def code2words(col: pd.Series, code_base: pd.DataFrame):
    """
    Transform the "code" column to their names
    Function search for the code in classifier base and return its name as string

    args:
        col: (pd.Series) column with code classifiers
        code_base: (pd.DataFrame) data base, that contains all codes and their corresponding names ('Код', 'Названия')
    
    return:
        pd.Series: code names as string
    """
    splitted_codes = col.str.split(";")
    splitted_codes = splitted_codes.parallel_apply(lambda row: list(set(row)))  # Get rid of dublicates
    splitted_codes = splitted_codes.explode()  # unzip list of codes
    words = splitted_codes.parallel_apply(lambda x: find_code_name_in_dict(x, code_base=code_base))
    words = words.groupby(words.index).apply(lambda x: " ".join(x))  # join all words in one sentence
    return words

def words2vectors(col: pd.Series):
    """Transform text columns to vectors and return vectors as the dataframe"""
    vectors = col.parallel_apply(vectorize)
    return pd.DataFrame(vectors.to_list(), columns=np.arange(0, 300))
