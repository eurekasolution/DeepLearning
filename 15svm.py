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

classifier = SVC(C=0.5)
training_points = [[1,2], [1,5], [2,2], [7,5], [9,4], [8,2]]
labels = [1,1,1,0,0,0]
classifier.fit(training_points, labels)
print("classifier.predict(3,2) :", classifier.predict([[3,2]]))