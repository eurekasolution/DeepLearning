#05tokenizer.py
# var lang="C,C++,JAVA,Python"

from konlpy.tag import Okt
tagger = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

sentences = tokenize(u'이것도 되나욕 ㅋㅋㅋ사릉해요')
print('Result : ', sentences)