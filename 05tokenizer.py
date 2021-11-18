#05tokenizer.py
# var lang="C,C++,JAVA,Python"

from konlpy.tag import Okt
tagger = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

sentences = tokenize(u'이것도 되나욕 ㅋㅋㅋ 사릉해요 나는 당신을 사랑합니다. 나는 딥러닝을 공부하고 있습니다. 딥러닝 재미있다.')
print('Result : ', sentences)

print("명사, 형용사, 동사만 추출")
nav_list = [word.split("/")[0] for word in sentences if
            word.split("/")[1] == "Noun" or
            word.split("/")[1] == "Adjective" or
            word.split("/")[1] == "Verb"
            ]
print(nav_list)