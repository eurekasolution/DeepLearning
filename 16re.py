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

print("type(x_train) = ", type(x_train))
print(x_train[0:10])