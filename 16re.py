#16re.py (정규식을 이용한 데이터 튜닝 후 학습)

from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt

import re
from tqdm import tqdm
from pprint import pprint
import nltk

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# debug, warning, info, error, danger, critical (7)

print("01... Data Loading")
train = pd.read_table("d:/ai/simple_train.txt")
test = pd.read_table("d:/ai/simple_test.txt")

print("02... 데이터 확인")
print("data type : ", type(train))
print("# of data : ", len(train))
print(train[0:10])

print("03... 데이터 분포 확인")
print(train.groupby('label').size().reset_index(name='count'))
print("Unique train : ", train['document'].nunique())
print("중복 제거 :")
train.drop_duplicates(subset=['document'], inplace=True)
print("중복 제거후 데이터수 :", len(train))
print("빈 데이터 수 : ",train.isnull().sum())
print("빈 데이터 있으면 삭제")
train = train.dropna(how='any');

print("04... 정규식 이용해 정제") # 한글, 공백제외한 나머지 삭제 a->""
train['document'] = train['document'].str.replace("[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", regex=True)
print("정규식 제거후 데이터 :", train[0:10])
# S    가나다
# S가나    다 , 문장이 시작하자 마자 공백이 있으면 제거
train['document'] = train['document'].str.replace("^ +", "", regex=True)
train['document'] = train['document'].str.replace("  ", "", regex=True)
train['document'].replace('', np.nan, inplace=True)
train = train.dropna(how='any')

test['document'] = test['document'].str.replace("[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", regex=True)
test = test.dropna(how='any');
# S    가나다
# S가나    다 , 문장이 시작하자 마자 공백이 있으면 제거
test['document'] = test['document'].str.replace("^ +", "", regex=True)
test['document'] = test['document'].str.replace("  ", "", regex=True)
test['document'].replace('', np.nan, inplace=True)
test = test.dropna(how='any')
print("정규식 제거후 데이터(Test) :", test[0:10])

print("05... Tokenize") # 안중근 의사, "의 "
# 학교 운동장에 학생(들)이 많이 있다.
# There are students in ...
stopwords = ['의', '가', '은', '는', '이', '들', '도', '좀', '잘', '걍', '으로', '로', '에', '와', '한', '하다']
okt = Okt()

x_train = []
for sentence in tqdm(train['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [ word for word in tokenized_sentence if not word in stopwords]
    x_train.append(stopwords_removed_sentence)

x_test = []
for sentence in tqdm(test['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [ word for word in tokenized_sentence if not word in stopwords]
    x_test.append(stopwords_removed_sentence)

print("type(x_train) = ", type(x_train))
print(x_train[0:10])

print("06... Embedding")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
print("토큰 워드 인덱스 : ", tokenizer.word_index)

threshold = 2 # 3, 1
total_cnt = len(tokenizer.word_index)
print("total cnt = ", total_cnt)
rare_cnt = 0
total_frequency = 0
rare_frequency = 0

for key, value in tokenizer.word_counts.items():
    total_frequency = total_frequency + value

    if(value < threshold):
        rare_cnt = rare_cnt +1
        rare_frequency = rare_frequency + value

print("[단어수] : ", total_cnt)
print("출현빈도 %s 번 이하인 단어수 : %s"%(threshold -1, rare_cnt))
print("희귀 집합 단어 비율 = {:.2f}%".format((rare_cnt / total_cnt) * 100))

print("[출현빈도 %s 미만 데이터 제외]"%(threshold))
voca_size = total_cnt - rare_cnt +1  # 0번은 제목
print("[단어 집합의 크기] : " , voca_size)

print("[Convert from Text Sequence to Integer Sequence]")
print("[Before] x_train \n",x_train[:3])
tokenizer = Tokenizer(voca_size)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print("[x_train type ] : ", type(x_train))
print("[After] x_train \n",x_train[:3])

y_train = np.array(train['label'])
y_test = np.array(test['label'])

print("[07] Remove Empty Samples : 출현빈도 적은 데이터로만 된 데이터는 빈 데이터!!!")
drop_train = [ index for index, sentence in enumerate(x_train) if len(sentence) <1]
print("[빈 샘플 삭제 수행]")
x_train = np.delete(x_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

print("[x_train, y_train 길이 검사] ", len(x_train), " , ", len(y_train))

print("[08] Padding - 서로 다른 길이 샘플 길이 동일하게!!")
print("[리뷰 최대 길이] : ", max(len(length) for length in x_train))
print("[리뷰 평균 길이] : {:.2f}".format(sum(map(len, x_train))/len(x_train)) )

print("[분포도] - Histogram")
#plt.hist([len(s) for s in x_train], bins=50)
#plt.xlabel("Length of Samples")
#plt.ylabel("Number of Samples")
#plt.show()

#
print("[최적의 max_len] - 대부분의 리뷰가 잘리지 않도록 하는 최적의 길이")
def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print("[전체 샘플 중 길이가 {:d} 이하인 샘플 비율 {:.2f} %".format(max_len, (count/len(nested_list))*100))

max_len = 40
below_threshold_len(max_len, x_train)
print("[최적의 max_len] 구할 때는 이렇게 해야한다.")
for i in range(5, 100, 5):
    below_threshold_len(i, x_train)

print("[Padding] max_len = ", max_len)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print("[09] Modeling ")
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(voca_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

print("[검증 데이터 손실 증가시 학습종료]")
# 검증 데이터 손실 4회 증가하면 학습 조기종료
# 검증 데이터 정확도(val_acc)가 이전보다 좋아질 경우에만 모델 저장

MODEL_SAVE_DIR_PATH = "d:/ai/model/"
if not os.path.exists(MODEL_SAVE_DIR_PATH):
    os.mkdir(MODEL_SAVE_DIR_PATH)

model_path = MODEL_SAVE_DIR_PATH + '{epoch:02d}-{val_loss:.04f}.hdf5'

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='max', verbos=1, save_best_only=False)

print("[10] 에포크 15회 수행, 훈련데이터중 20%를 검증 데이터로 사용해 정확도 확인")
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15,
                    callbacks=[es, mc],
                    batch_size=64,
                    validation_split=0.2)

model.save("d:/ai/model/my_model")
print("\nAccuracy: {:.4f}".format(model.evaluate(x_train, y_train)[1]))
print("[11] 정확도 측정")
loaded_model = load_model("d:/ai/model/my_model")
print("\n[테스트 정확도] : {:.4f} %".format((loaded_model.evaluate(x_test, y_test)[1])))

print("[12] 리뷰 예측")
def sentiment_predict(new_sentence):
    org_sentence = new_sentence
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True)      # Tokenize
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)        # Padding
    score = float(loaded_model.predict(pad_new))            # Predict
    if(score > 0.5):
        print("[{}] 긍정 리뷰 확률 {:.2f}%".format(org_sentence, score * 100))
    else:
        print("[{}] 부정 리뷰 확률 {:.2f}%".format(org_sentence, (1-score) * 100))

print("[리뷰 예측 테스트]")
sentiment_predict("주인공 연기가 진짜 대단하다. ㅠㅠ") # 주인공 연기 진짜 대단하다 ㅠㅠ
sentiment_predict("시간 남아도는 사람만 이 영화 보세요") # 시간 남아도 사람만 이 영화 보세요
sentiment_predict("각오하고 보니까 의외로 재미는 있다. 멋져버려")
sentiment_predict("이건 아니지")
sentiment_predict("다시 또 보고 싶은 영화")
sentiment_predict("재미있을까? 아직은 모르겠다.")
sentiment_predict("ㅠㅠ")
sentiment_predict("웃겨 죽는줄 알았다")
sentiment_predict("부모님과는 보면 안될것 같은 영화")
sentiment_predict("오~~")
sentiment_predict("잘하는 짓이다")






