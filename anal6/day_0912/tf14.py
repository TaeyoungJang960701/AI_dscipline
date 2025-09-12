import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Model

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv')
print(data.head(2))
del data['no']
housing = fetch_california_housing()

# train / test
x_train_all,x_test,y_train_all,y_test=train_test_split(housing.data,housing.target,test_size=0.2,random_state=12)
print(x_train_all.shape,x_test.shape,y_train_all.shape,y_test.shape)

# train : train / validation
x_train,x_valid,y_train,y_valid=train_test_split(x_train_all,y_train_all,test_size=0.3,random_state=12)
print(x_train_all.shape,x_valid.shape,y_train_all.shape,y_valid.shape)

print(x_train[:3])
print()

# 표준화
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_valid=scaler.fit_transform(x_valid)
x_test=scaler.fit_transform(x_test)
print(x_train[:3])

print('Sequential Api --단순한 방법으로 MLP---')      # MultiLayer Perceptron
model = Sequential()
model.add(Input(shape = x_train.shape[1:]))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))
model.summary()

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
history = model.fit(x_train, y_train, epochs = 20, validation_data = (x_valid, y_valid), verbose = 2)

print('evaluate : ', model.evaluate(x_test, y_test, verbose = 0))

# test 일부 자료로 예측
x_new = x_test[:3]
y_pred = model.predict(x_new)
print('예측값 : ', y_pred.ravel())
print('실제값 : ', y_test[:3])

plt.plot(range(1, 21), history.history['mse'], c = 'b', label = 'mse')
plt.plot(range(1, 21), history.history['val_mse'], c = 'r', label = 'val_mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.show()

print('functional api --유연한 MLP ---')

# 이게 하나의 층
input_ = Input(shape = x_train.shape[1:])
net1 = Dense(units = 32, activation = 'relu')(input_)
net2 = Dense(units = 32, activation = 'relu')(net1)

# 이것도 하나의 층
concat = Concatenate()([input_, net2])
output = Dense(units = 1)(concat)

model2 = Model(inputs = [input_], outputs = [output])

model2.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
history = model2.fit(x_train, y_train, epochs = 20, validation_data = (x_valid, y_valid), verbose = 2)

print('evaluate : ', model2.evaluate(x_test, y_test, verbose = 0))

# test 일부 자료로 예측
x_new = x_test[:3]
y_pred = model2.predict(x_new)
print('예측값 : ', y_pred.ravel())
print('실제값 : ', y_test[:3])

plt.plot(range(1, 21), history.history['mse'], c = 'b', label = 'mse')
plt.plot(range(1, 21), history.history['val_mse'], c = 'r', label = 'val_mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.show()
plt.close()

print('functional api 2 -- 일부 특성은 짧은 경로로 전달, 다른 특성은 깊은 경로로 전달 MLP ---')
# 예: 앞 5개 특성은 짧은 경로, 뒤 6개 특성은 깊은 경로
input_a = Input(shape=[5], name='wide_input')
input_b = Input(shape=[3], name='deep_input') # Changed shape to [3]
net1 = Dense(units=32, activation='relu')(input_b)
net2 = Dense(units=32, activation='relu')(net1)
concat = Concatenate()([input_a, net2])   # ← 변수명 수정
output = Dense(units=1, name='output')(concat)

model3 = Model(inputs=[input_a, input_b], outputs=[output])

model3.compile(optimizer='adam', loss='mse', metrics=['mse'])

# 데이터 분리
x_train_a, x_train_b = x_train[:, :5], x_train[:, 5:] # Changed slicing for x_train_b
x_valid_a, x_valid_b = x_valid[:, :5], x_valid[:, 5:] # Changed slicing for x_valid_b
x_test_a,  x_test_b  = x_test[:, :5],  x_test[:, 5:]  # Changed slicing for x_test_b
x_new_a,   x_new_b   = x_test_a[:3],   x_test_b[:3]

# 학습
history3 = model3.fit([x_train_a, x_train_b], y_train,
                      epochs=20,
                      validation_data=([x_valid_a, x_valid_b], y_valid),
                      verbose=2)

# 평가
print('evaluate3 : ', model3.evaluate([x_test_a, x_test_b], y_test, verbose=0))

# test 일부 자료로 예측
y_pred = model3.predict([x_new_a, x_new_b])
print('예측값 : ', y_pred.ravel())
print('실제값 : ', y_test[:3])

# 학습曲선
plt.plot(range(1, 21), history3.history['mse'], c='b', label='mse')
plt.plot(range(1, 21), history3.history['val_mse'], c='r', label='val_mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.show()

