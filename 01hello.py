print("-"*80)

# comment
# {} 없다

from konlpy.tag import Kkma

kkma = Kkma()
print('kkma 문장 분리 :  ', kkma.sentences(u'안녕하세요. 홍길동입니다. 반갑습니다. 저는 인공지능입니다.'))
print('kkma 문장 분리 :  ', kkma.sentences(u'깃허브 테스트를 수행하고 있습니다.'))

print("-"*80)
from konlpy.tag import Okt
tagger = Okt()

print('Okt 명사 추출 : ', tagger.nouns(u'테헤란로 주변 빌딩숲 사이에 있는 커피숍과 편의점 ㅠㅠ, ㅋㅋ 나는 너를 사릉해 '))
print('Okt 품사 추출 : ',  tagger.pos(u'우리나라 테헤란로 주변 빌딩숲 사이에 있는 커피숍과 편의점 ㅠㅠ, ㅋㅋ, 나는 너를 사랑해'))
print('Okt 오타 원형처리 : ',  tagger.pos(u'안녕하세용 난 잘못지내는뎅 밥한번먹읍시당'))
print('Okt 오타 원형처리 : ',  tagger.pos(u'안녕하세용 난 잘못지내는뎅 밥한번먹읍시당', norm=True, stem=True))