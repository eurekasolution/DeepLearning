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

