# 17news.py (뉴스 크롤링, 분석)
import codecs

positive = []
negative = []
posneg = []

with open("d:/ai/negative_words_self.txt", encoding='utf-8') as neg:
    negative = neg.readlines()
negative = [neg.replace("\n", "") for neg in negative]

with open("d:/ai/positive_words_self.txt", encoding='utf-8') as pos:
    positive = pos.readlines()

negative = [neg.replace("\n", "") for neg in negative]
positive = [pos.replace("\n", "") for pos in positive]

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm

df_labels = []
df_titles = []

for k in tqdm(range(100)):  # 1 당 10개의 뉴스를 크롤링한다. 1은 페이지 수
    num = k * 10 + 1
    # 삼성전자
    url = "https://search.naver.com/search.naver?&where=news&query=맘스터치&start=" + str(num)
    print("URL = ",url)
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    titles = soup.select("a.news_tit")
    print("titles = ",titles)

    for title in titles:
        #print("1. title = ",title)
        title_data = title.get_text()
        print("2. title_data : ",title_data)

        clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title_data)
        negative_flag = False
        print("3. clean title : ", clean_title)
        label = 0       # 중립(기본값)

        for i in range(len(negative)):
            if negative[i] in clean_title:
                label = -1
                negative_flag = True
                print("4. negative 비교 단어 : ", negative[i], "\nclean title : ", clean_title)
                break
        if negative_flag == False:
            for i in range(len(positive)):
                if positive[i] in clean_title:
                    label = 1
                    print("5. positive 비교 단어 : ", positive[i], "\nclean title : ", clean_title)
                    break
        df_titles.append(clean_title)
        df_labels.append(label)

my_dataframe = pd.DataFrame({"title":df_titles, "label":df_labels})
print("6. 데이터 확인")
print("----- titles = \n", df_titles)
print("----- labels = \n", df_labels)
print("----- df_title_df = \n", my_dataframe)

# save to csv
print("7. Save to CSV")
my_dataframe.to_csv("d:/ai/moms.csv", index=False, encoding="utf-8-sig", header=True)

print("End of Program !!!")
