#07ngram.py

def ngram(s, num):
    res=[]
    slen = len(s) - num + 1 # 나는 너를 사랑한다
    for i in range(slen):
        ss=s[i:i+num]
        res.append(ss)
    return res

def diff_ngram(sa, sb, num):
    a=ngram(sa, num)
    b=ngram(sb, num)
    r=[]
    cnt=0
    for i in a:
        for j in b:
            if i==j:
                cnt+=1
                r.append(i)
    return cnt/len(a), r

print("test 01")
a="오늘 강남에서 맛있는 스파게티를 먹었다. 오늘 기분 좋다"
b="강남에서 먹었던 오늘의 스파게티는 맛있었다. 오늘의 일기 끝"

print("bi-gram")
r2, word2=diff_ngram(a, b, 2)
print("bigram :", r2, word2)

print("trigram") ###
r3, word3=diff_ngram(a, b, 3)
print("trigram :", r3, word3)
