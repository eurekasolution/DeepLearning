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