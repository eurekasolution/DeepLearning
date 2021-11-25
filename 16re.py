#16re.py (정규식을 이용한 데이터 튜닝 후 학습)

from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt

import re
from tqdm import tqdm
from pprint import pprint
import nltk

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# debug, warning, info, error, danger, critical (7)

print("01... Data Loading")
train = pd.read_table("d:/ai/simple_train.txt")
test = pd.read_table("d:/ai/simple_test.txt")

print("02... 데이터 확인")
print("data type : ", type(train))
print("# of data : ", len(train))
print(train[0:10])

print("03... 데이터 분포 확인")
print(train.groupby('label').size().reset_index(name='count'))
print("Unique train : ", train['document'].nunique())
print("중복 제거 :")
train.drop_duplicates(subset=['document'], inplace=True)
print("중복 제거후 데이터수 :", len(train))
print("빈 데이터 수 : ",train.isnull().sum())
print("빈 데이터 있으면 삭제")
train = train.dropna(how='any');

print("04... 정규식 이용해 정제") # 한글, 공백제외한 나머지 삭제 a->""
train['document'] = train['document'].str.replace("[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", regex=True)
print("정규식 제거후 데이터 :", train[0:10])
# S    가나다
# S가나    다 , 문장이 시작하자 마자 공백이 있으면 제거
train['document'] = train['document'].str.replace("^ +", "", regex=True)
train['document'] = train['document'].str.replace("  ", "", regex=True)
train['document'].replace('', np.nan, inplace=True)
train = train.dropna(how='any')

test['document'] = test['document'].str.replace("[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", regex=True)
test = test.dropna(how='any');
# S    가나다
# S가나    다 , 문장이 시작하자 마자 공백이 있으면 제거
test['document'] = test['document'].str.replace("^ +", "", regex=True)
test['document'] = test['document'].str.replace("  ", "", regex=True)
test['document'].replace('', np.nan, inplace=True)
test = test.dropna(how='any')
print("정규식 제거후 데이터(Test) :", test[0:10])

print("05... Tokenize") # 안중근 의사, "의 "
# 학교 운동장에 학생(들)이 많이 있다.
# There are students in ...
stopwords = ['의', '가', '은', '는', '이', '들', '도', '좀', '잘', '걍', '으로', '로', '에', '와', '한', '하다']
okt = Okt()

x_train = []
for sentence in tqdm(train['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [ word for word in tokenized_sentence if not word in stopwords]
    x_train.append(stopwords_removed_sentence)

x_test = []
for sentence in tqdm(test['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [ word for word in tokenized_sentence if not word in stopwords]
    x_test.append(stopwords_removed_sentence)

print("type(x_train) = ", type(x_train))
print(x_train[0:10])

print("06... Embedding")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
print("토큰 워드 인덱스 : ", tokenizer.word_index)

threshold = 2 # 3, 1
total_cnt = len(tokenizer.word_index)
print("total cnt = ", total_cnt)
rare_cnt = 0
total_frequency = 0
rare_frequency = 0

for key, value in tokenizer.word_counts.items():
    total_frequency = total_frequency + value

    if(value < threshold):
        rare_cnt = rare_cnt +1
        rare_frequency = rare_frequency + value

print("[단어수] : ", total_cnt)
print("출현빈도 %s 번 이하인 단어수 : %s"%(threshold -1, rare_cnt))
print("희귀 집합 단어 비율 = {:.2f}%".format((rare_cnt / total_cnt) * 100))

print("[출현빈도 %s 미만 데이터 제외]"%(threshold))
voca_size = total_cnt - rare_cnt +1  # 0번은 제목
print("[단어 집합의 크기] : " , voca_size)

print("[Convert from Text Sequence to Integer Sequence]")
print("[Before] x_train \n",x_train[:3])
tokenizer = Tokenizer(voca_size)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print("[x_train type ] : ", type(x_train))
print("[After] x_train \n",x_train[:3])

y_train = np.array(train['label'])
y_test = np.array(test['label'])

print("[07] Remove Empty Samples : 출현빈도 적은 데이터로만 된 데이터는 빈 데이터!!!")
drop_train = [ index for index, sentence in enumerate(x_train) if len(sentence) <1]
print("[빈 샘플 삭제 수행]")
x_train = np.delete(x_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

