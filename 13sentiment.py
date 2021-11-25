#13sentiment.py

from tensorflow.keras.preprocessing.text import Tokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# debug, warning, info, error, danger, critical (7)

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        result = [line.split('\t') for line in f.read().splitlines()]
        result = result[1:]
    return result

print("01 ... read file")
train = read_data("d:/ai/ratings_train.txt")
test = read_data("d:/ai/ratings_test.txt")

print("02 ... word Embdding")
def korean_movie(max_num_words=1000):
    train_x = []
    train_y = []

    for i in range(len(train)):
        train_x.append(train[i][1]) # 댓글 데이터
        train_y.append(int(train[i][2])) # 긍정(1), 부정(0)

    test_x = []
    test_y = []

    for i in range(len(test)):
        test_x.append(test[i][1])  # 댓글 데이터
        test_y.append(int(test[i][2]))  # 긍정(1), 부정(0)

    # embedding
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(train_x)

    token_train_x = tokenizer.texts_to_sequences(train_x)
    token_test_x = tokenizer.texts_to_sequences(test_x)

    return (token_train_x, train_y), (token_test_x, test_y)

print("03 ... Analysis")
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.datasets import imdb

print("04 ... Modeling")
max_num_words = 5000
maxlen = 100    # 문장의 최대 길이
batch_size = 32

(x_train, y_train), (x_test, y_test) = korean_movie(max_num_words)

print("04-1 ... Check Data")
print(x_train[0:5])
print(y_train[0:5])

print("05 ... 단어길이 조정(100)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train[0:5])
print(y_train[0:5])

print("06 ... Modeling")
model = Sequential()
model.add(Embedding(max_num_words, 128))
model.add(LSTM(128,
               dropout=0.2,
               recurrent_dropout=0.2 ,
               input_shape=(3,1)))
model.add(Dense(1, activation='sigmoid'))

print("07 ... Compiling ")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("08 ... Learning")
import numpy as np
y_train = np.array(y_train)
y_test = np.array(y_test)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
#test
print("09 ... Evaluation")
loss, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print("loss : ", loss)
print("acc : ", acc)
