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
