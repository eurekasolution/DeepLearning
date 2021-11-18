#10tuning.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
z = np.array([11,12,13,14,15])


model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuacy'])
model.fit(x, y, epochs=100,batch_size=1)

mse = model.evaluate(x, y, batch_size=1)
print('mse : ' , mse)

loss, acc=model.evaluate(x,y, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

predict = model.predict(z)
print(predict)