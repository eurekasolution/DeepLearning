print("-"*80)

# comment
# {} 없다

from konlpy.tag import Kkma

kkma = Kkma()
print('kkma 문장 분리 :  ', kkma.sentences(u'안녕하세요. 홍길동입니다. 반갑습니다. 저는 인공지능입니다.'))
#addeda