#02wordcloud.py
# pip install wordcloud
# pip install matplotlib
# pip install gensim

from gensim import corpora
from gensim import models
documents=[
    '나는 아침에 라면을 자주 먹는다.',
    '나는 아침에 밥 대신에 라면을 자주 먹는다.',
    '현대인의 삶에서 스마트폰은 필수품이 되었다.',
    '현대인들 중에서 스마트폰을! 사용하지 않는 사람은 거의 없다.',
    '점심시간에 스마프폰을 이용해 영어? 회화 공부를 하느라 혼자 밥을 먹는다.'
]

# Stop word
stopWordList = ('.,;!?')
texts = [
            [word for word in document.split()
                if word not in stopWordList]
            for document in documents
]
print("-"*80)
print(documents)

print("-"*80)
print(texts)

print("make Dictionary")
dictionary = corpora.Dictionary(texts)
print("make corpus") # 말뭉치, 단어사전
corpus = [dictionary.doc2bow(text) for text in texts]
print('corpus : {}'.format(corpus))

