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
tagger = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

docs = []
for row in ratings_train:
    docs.append(row[1])

data = [tokenize(d) for d in docs]

print("Creating model")
model = word2vec.Word2Vec(data)

model.init_sims(replace=True)

# 남자 + 배우 - 여배우
pprint(model.wv.most_similar(positive=tokenize(u'남자 배우'),
                             negative=tokenize(u'여배우'),
                             topn=10))