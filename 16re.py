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