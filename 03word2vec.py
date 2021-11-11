# word2vec.py

from gensim.models import word2vec
import logging
import sys

print("time")
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print("read file from parameter")  # ./03word2vec.py a.txt b.txt
sentences = word2vec.LineSentence(sys.argv[1])

model = word2vec.Word2Vec(sentences,
                          vector_size = 100,
                          min_count=1,
                          window=10)
model.save(sys.argv[2])