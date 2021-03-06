# 15svm.py (감정분석)

import nltk
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.svm import LinearSVC, SVC
from nltk.classify import SklearnClassifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        result = [line.split('\t') for line in f.read().splitlines()]
        result = result[1:]
    return result

print("01... 데이터 파일 읽기")
train = read_data("d:/ai/simple_train.txt") # simple_train.txt (15만개->1만개)
test = read_data("d:/ai/simple_test.txt")   # simple_test.txt

print("02... SVM 간단한 코드 사용방법")
classifier = SVC(C=0.5)
training_points = [[1,2], [1,5], [2,2], [7,5], [9,4], [8,2]]
labels = [1,1,1,0,0,0]
classifier.fit(training_points, labels)
print("classifier.predict(3,2) :", classifier.predict([[3,2]]))

print("03... 데이터 확인")
print("length train : ", len(train)) # 5000
print("length test : ", len(test)) # 1000
print("length train[0] : ", len(train[0]))

print("04... Tokenize")
from konlpy.tag import Okt
import json
from pprint import pprint

okt = Okt()
def tokenizing(docs):
    return ['/'.join(t) for t in okt.pos(docs, norm=True, stem=True)]

train_docs = [ (tokenizing(row[1]), row[2]) for row in train]
test_docs = [ (tokenizing(row[1]), row[2]) for row in test]

tokens = [ t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')
print("text[0:5] : ", text[0:5])
print("Top 10 : ", text.vocab().most_common(10))

print("05... Classify")
selected_words = [f[0] for f in text.vocab().most_common(300)]
def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

train_docs = train_docs[:1000]
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

print("06... Evaluate")
classifier = SklearnClassifier(LinearSVC())
classifier_svm = classifier.train(train_xy)
print(nltk.classify.accuracy(classifier_svm, test_xy))

# 정규식 (Regular Expression, RE, PCRE)
# 패턴을 검사
# [a-z], [a-zA-Z], [a-z\/]
# 구둣점, !, #, $  : 제거
# 가나다, ㅋㅋ, ㅠㅠ
# 예: 아, 이 영화 !! 정말 짜증난다~~~~ ㅠㅠ
# 필요한 단어: 아 이 영화 정말 짜증난다
# [^a-zA-Z]
# [^가-힣ㄱ-ㅎㅏ-ㅣ]
