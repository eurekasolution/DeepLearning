#12validation.py

import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])

z = np.array([21, 22, 23, 24, 25])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train,
          epochs=100,
          batch_size=1,
          validation_data=(x_val, y_val))

loss, acc=model.evaluate(x_test,y_test,
                         batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

predict = model.predict(z)
print(predict)