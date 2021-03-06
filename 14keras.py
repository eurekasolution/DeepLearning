# 14keras.py (감정분석, 데이터분포, 처리과정 확인)

from tensorflow.keras.preprocessing.text import Tokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# debug, warning, info, error, danger, critical (7)

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        result = [line.split('\t') for line in f.read().splitlines()]
        result = result[1:]
    return result
# 데이터의 예
#id	document	label
#9976970	아 더빙.. 진짜 짜증나네요 목소리	0
#3819312	흠...포스보고 초딩영화줄....오버연기조차 가볍지 않구나	1
#10265843	터너무재밓었다그래서보는것을추천한다	0

print("01... 데이터 파일 읽기")
train = read_data("d:/ai/simple_train.txt") # simple_train.txt (15만개->1만개)
test = read_data("d:/ai/simple_test.txt")   # simple_test.txt

print("02... Tokenize")
from konlpy.tag import Okt
import json
from pprint import pprint
# JavaScript Object Notation
# key:"hello"

okt = Okt()
okt.pos("나는 딥러닝을 배우고 있습니다. 딥러닝은 재미있는데 어렵습니다.")

def tokenizing(docs):
    return ['/'.join(t) for t in okt.pos(docs, norm=True, stem=True)]

train_pos = []
test_pos = []

for row in train:
    try:
        train_pos0 =[tokenizing(row[1]), row[2]]
        train_pos.append(train_pos0)
    except:
        pass

for row in test:
    try:
        test_pos0 =[tokenizing(row[1]), row[2]]
        test_pos.append(test_pos0)
    except:
        pass

pprint(train_pos[0])
# [ ['아/감탄사', '더빙/명사', '../구둣점'] , [ ...], [... ] ]
print("03... 데이터 전처리")
tokens = [t for d in train_pos for t in d[0]]

import nltk
text = nltk.Text(tokens, name='NMSC')
print("Length of text token : ", len(set(text.tokens))) # A,A, A, B, B, C => A, B, C
text.vocab().most_common(10)
print("top 10 list :", text.vocab().most_common(10)) # . 구둣점, 영화/명사

print("04... 데이터 분포 확인")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name ="C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_name).get_name()
#plt.rc('font', family=font_name)
#plt.figure(figsize=(20, 20))
#text.plot(50)

print("05.. 단어 수를 제한")
selected_words = [f[0] for f in text.vocab().most_common(1000)]

print("06... 데이터 출현 빈도별 분포")
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_pos]
test_x = [term_frequency(d) for d, _ in test_pos]
train_y = [c for _, c in train_pos]
test_y = [c for _, c in test_pos]

print("-"*80)
print("length train_x : ", len(train_x))
print("train_x[0:10] : ", train_x[0:10])
print("length train_y : ", len(train_y))

import numpy as np

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

print("07 ... Modeling")
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.datasets import imdb

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1000, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print("08... Compile")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("09... Learning")
model.fit(x_train,
          y_train,
          batch_size=512,
          epochs=5,     # 10, 15
          validation_data=(x_test, y_test))

loss, acc=model.evaluate(x_test, y_test)
print("Loss : ", loss)
print("Acc : ", acc)

def predict_pos_text(text):
    token = tokenizing(text)
    freq = term_frequency(token)

    data = np.expand_dims(np.array(freq).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score >0.5):
        print("[{}] 는 {:.2f}% 확률로 긍정 리뷰입니다.".format(text, score * 100))
    else:
        print("[{}] 는 {:.2f}% 확률로 부정 리뷰입니다.".format(text, score * 100))

print("10... Predict")
predict_pos_text("와 이 영화 정말 재미있다.")
predict_pos_text("와 이 영화 정말 짜증있다.")
predict_pos_text("ㅠㅠ 이 영화 ㅠㅠ")
predict_pos_text("너무 웃겨")
predict_pos_text("너무 슬퍼")
predict_pos_text("오~~~")

print("-"*80)
print("11... Predict from File")
ratings_train = read_data("d:/ai/review.txt")

for row in ratings_train[0:100]:
    predict_pos_text(row[1])