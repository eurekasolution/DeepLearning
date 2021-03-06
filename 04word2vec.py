# 04 word2vec.py
from gensim.models import Word2Vec
import sys

model = Word2Vec.load(sys.argv[1]) #word2vec.py d:/ai/text8.model
results = model.wv.most_similar(positive=['woman', 'king'],
                                negative=['man'],
                                topn=10)

print("result : woman + king - man")
for result in results:
    print(result[0], '\t', result[1])

# 터미널에서 명령
#  python 04word2vec.py d:/ai/text8.model
# 크롤링..댓글, 영화 댓글,,