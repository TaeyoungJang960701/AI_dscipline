# 문제2)
# https://github.com/pykwon/python/tree/master/data
# 자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
# 모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.
# 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력하시오.

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv')
print(data.head())
print(data.columns)
data.drop(['datetime'], axis = 1, inplace = True)

# print(data.corr())      # 결과 count와 registered의 상관계수가 가장 높아보임 플러스 알파로 casual 쓰자
# print(data.info())
# data = data.dropna()
print(data.isna().sum())

sns.pairplot(data[['count', 'registered','casual']])
plt.show()
plt.close()

# data = data[['count','registered','casual']]
x = data[['casual']]
# x = data[['registered', 'casual']]

y = data[['count']]

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)
print(x_train_all.shape, x_test.shape, y_train_all.shape, y_test.shape)

# train_data = data.sample(frac = 0.7, random_state = 42)
# test_data = data.drop(train_data.index)

# print('train 데이터 : ', train_data[:3], train_data.shape)  # (7620, 11)
# print('test 데이터 : ', test_data[:3], test_data.shape)     # (3266, 11)

# train_stat = train_data.describe()
# print(train_stat)

x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size = 0.2, random_state = 1)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

print(x_train[:3])
print('-' * 100)

# 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.fit_transform(x_valid)
x_test = scaler.fit_transform(x_test)
print(x_train[:3])

print('Sequential Api를 써보자')    # MultiLayer Perceptron
model = Sequential()
model.add(Input(shape = x_train.shape[1:]))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))
model.summary()

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
history = model.fit(x_train, y_train, epochs = 100, validation_data = (x_valid, y_valid), verbose = 2)

print('evaluate : ', model.evaluate(x_test, y_test, verbose = 0))

# test로 예측
x_new = x_test[:3]
y_pred = model.predict(x_new)
print('예측값 : ', y_pred.ravel())
print('실제값 : ', y_test[:3])

plt.plot(range(len(history.history['mse'])), history.history['mse'], c = 'b', label = 'mse')
plt.plot(range(len(history.history['val_mse'])), history.history['val_mse'], c = 'r', label = 'val_mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.show()
plt.close()

