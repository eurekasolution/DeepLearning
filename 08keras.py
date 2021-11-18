#09keras.py
# 0. 사용할 패키지 불러오기
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

print("1. ... DataSet ...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("2. ... Modeling ...")
model = Sequential()
model.add(Dense(units=64, input_dim=28 * 28, activation='relu'))  # O O O O O O ..... O O O O
model.add(Dense(units=10, activation='softmax'))                  # O O O O ... O O

print("3. ... Config Model Learning ...")
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print("4. ... Learning ...")
hist = model.fit(x_train, y_train, epochs=5, batch_size=32)

print("5. ... Review Learning ...")
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['accuracy'])

print("6. Evaluate Model ...")
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

print("7. ... Use Modeling ...")
xhat = x_test[0:1]
yhat = model.predict(xhat)
print('## yhat ##')
