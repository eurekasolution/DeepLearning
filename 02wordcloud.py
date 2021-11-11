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

print("-"*80)
print("create model")
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=1)
print("sorting")
for t in lda.show_topics():
    print(t)

print("visulization")
import matplotlib.pyplot as plt
from wordcloud import  WordCloud

wc = WordCloud(background_color='white', font_path='C:/Windows/Fonts/malgun.ttf')

plt.figure(figsize=(30,30))
for t in range(lda.num_topics):
    plt.subplot(4,4, t+1)
    x = dict(lda.show_topic(t, 200))
    im = wc.generate_from_frequencies(x)
    plt.imshow(im)
    plt.title("Topic # " + str(t))

# save file
plt.savefig('d:/ai/wc.png', bbox_inches='tight')
