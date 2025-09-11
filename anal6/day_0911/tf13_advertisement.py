import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv')
print(data.head(2))
del data['no']

fdata = data[['tv','radio','newspaper']]
ldata = data[['sales']]
print(fdata[:2])
print(ldata[:2])

# 정규화      이건 그냥 쌤이 한번 해본거야
# scaler = MinMaxScaler(feature_range = (0,1))
# fedata = scaler.fit_transform(fdata)
# print(fedata[:3])

fedata = minmax_scale(fdata, axis = 0, copy = True)   # 원본 보존
print(fedata[:3])

# train / test
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata, test_size = 0.3, random_state = 123)

model = Sequential()
model.add(Input(shape = (3,)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])

# print(model.summary())
# tf.keras.utils.plot_model(model, 'tf13.png')

history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, verbose = 2, validation_split = 0.2)
# validation_data = (x_vali, y_vali)

# 모델 평가 점수
loss = model.evaluate(x_test, y_test, verbose = 0)
print('loss : ', loss[0])

# 모델 평가 점수
loss = model.evaluate(x_test, y_test, verbose = 0)
print('loss : ', loss[0])

# history 값 확인
print('history : ', history.history)
print('loss : ', history.history['loss'])
print('mse : ', history.history['mse'])
print('val_loss : ', history.history['val_loss'])
print('val_mse : ', history.history['val_mse'])

# loss 시각화
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()
plt.close()

print('r2score : ', r2_score(y_test, model.predict(x_test)))

