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
