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
#3819312	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나	1
#10265843	너무재밓었다그래서보는것을추천한다	0

print("01... 데이터 파일 읽기")
train = read_data("d:/ai/ratings_train.txt")
test = read_data("d:/ai/ratings_test.txt")

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
        train_pos0 =[tokenizing(row[1], row[2])]
        train_pos.append(train_pos0)
    except:
        pass
