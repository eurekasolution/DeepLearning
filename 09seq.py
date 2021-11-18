#09seq.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')  ## <-- loss 오타 났었습니다.
model.fit(x, y, epochs=100,batch_size=1)

mse = model.evaluate(x, y, batch_size=1)
print('mse : ' , mse)

# 할일
# 1. 위 14라인 오타 수정
# 2. 에러 해결하기
# 에러 메시지를 보면
# Microsoft C++ Redistributable for Visual Studio 2015, 2017 and 2019"
# 재배포 패키지가 설치되어야 합니다.
# 배포해 드렸던 파일에 보면
# 02-VC_redist.x64 가 있습니다. 이 글 보시면 설치해 주시고,
# 설치후 컴퓨터가 Reboot이 필요합니다.

