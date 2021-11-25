# 18news.py
#  news_test, news_train 데이터로 학습한 후에, hansol.csv파일을 예측

import pandas as pd

train_data = pd.read_csv("d:/ai/news_train.csv", encoding="utf-8")
test_data = pd.read_csv("d:/ai/news_test.csv", encoding="utf-8")

import matplotlib.pyplot as plt
print("Step 1. 데이터 분포 분석")
print("1-1. 분포도 확인")
#train_data['label'].value_counts().plot(kind='bar')

print("1-2. 분포도 숫자로 확인")
print("Train Data\n",train_data.groupby('label').size().reset_index(name='count'))
print("Test Data\n",test_data.groupby('label').size().reset_index(name='count'))

print("Step 2. 모델만들기 전처리")
print("2-1. 토큰화")
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '에', '와', '한', '하다']

import konlpy
from konlpy.tag import Okt

okt = Okt()
x_train = []
for sentence in train_data['title']:
    temp_x = []
    temp_x = okt.morphs(sentence, stem=True)
    temp_x = [word for word in temp_x if not word in stopwords]
    x_train.append(temp_x)

x_test = []
for sentence in test_data['title']:
    temp_x = []
    temp_x = okt.morphs(sentence, stem=True)
    temp_x = [word for word in temp_x if not word in stopwords]
    x_test.append(temp_x)

print("3. 토근화 확인")
print("3-1. Train Data\n", x_train[:5])
print("3-2. Test Data\n", x_test[:5])

print("Step 3. 정수 인코딩")
from tensorflow.keras.preprocessing.text import Tokenizer
max_words = 50000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print("4. 정수화 데이터 확인")
print("4-1. Train Data\n", x_train[:5])
print("4-2. Test Data\n", x_test[:5])


