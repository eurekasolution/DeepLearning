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
max_words = 1000 # 50000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print("4. 정수화 데이터 확인")
print("4-1. Train Data\n", x_train[:5])
print("4-2. Test Data\n", x_test[:5])

print("5. 패딩을 위한 최대길이, 평균길이 분포 확인")
print("5-1 제목 최대 길이(Train Data) : ", max(len(length) for length in x_train))
print("5-2 제목 평균 길이(Train Data) : {:.2f}".format(sum(map(len, x_train))/len(x_train)))

print("5-3 길이 분포도(Train Data)")
#plt.hist([len(s) for s in x_train], bins=50)
#plt.xlabel("Length of Data")
#plt.ylabel("# of Data")
#plt.show()

print("5-4 제목 최대 길이(Test Data) : ", max(len(length) for length in x_test))
print("5-5 제목 평균 길이(Test Data) : {:.2f}".format(sum(map(len, x_test))/len(x_test)))
print("5-6 길이 분포도(Test Data)")
#plt.hist([len(s) for s in x_test], bins=50)
#plt.xlabel("Length of Data")
#plt.ylabel("# of Data")
#plt.show()

print("5-7 One-Hot Encoding")

import numpy as np

y_train = []
y_test = []

for i in range(len(train_data['label'])):
    if train_data['label'].iloc[i] == 1:
        y_train.append([1, 0, 0])
    elif train_data['label'].iloc[i] == 0:
        y_train.append([0, 1, 0])
    elif train_data['label'].iloc[i] == -1:
        y_train.append([0, 0, 1])

for i in range(len(test_data['label'])):
    if test_data['label'].iloc[i] == 1:
        y_test.append([1, 0, 0])
    elif test_data['label'].iloc[i] == 0:
        y_test.append([0, 1, 0])
    elif test_data['label'].iloc[i] == -1:
        y_test.append([0, 0, 1])

y_train = np.array(y_train)
y_test = np.array(y_test)
print("5-8 y_train\n",y_train)
print("5-9 y_test\n",y_test)

print("Step 4. 모델 만들기")
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 20
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',  #optimizer='rmsprop'
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

print("\n6. 테스트 정확도 : {:.2f}%".format(model.evaluate(x_test, y_test)[1] * 100))

print("Step 5. 예측")
predict = model.predict(x_test)

predict_labels = np.argmax(predict, axis=1)
original_labels = np.argmax(y_test, axis=1)

for i in range(30):
    print("기사제목 : ", test_data['title'].iloc[i], "\t원라벨 : ", original_labels[i], "\t예측값 :", predict_labels[i])

