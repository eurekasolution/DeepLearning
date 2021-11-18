#11dataset.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

z = np.array([111,112,113,114,115])


model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100,batch_size=1)

loss, acc=model.evaluate(x_test,y_test, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

predict = model.predict(z)
print(predict)