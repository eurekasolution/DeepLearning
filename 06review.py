#06review.py

import codecs
from konlpy.tag import Okt
from gensim.models import word2vec
from konlpy.utils import pprint

def read_data(filename): # d:/ai/review.txt
    with codecs.open(filename, encodingn='utf-8', mode='r') as f:
        data = [ line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

ratings_train = read_data("d:/ai/review.txt")