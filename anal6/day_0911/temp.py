# data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
#  - train / test 분리
#  - Sequential api와 function api 를 사용해 모델을 만들어 보시오.
#  - train과 test의 mse를 시각화 하시오
#  - 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('https://raw.githubusercontent.com/data-8/materials-fa17/refs/heads/master/lec/galton.csv')
# print(df.head())

df.drop(df[df['gender'] != 'male'].index, inplace = True)
print(df.head())
print('-' * 100)

df_father = df[['father']]      # 이게 feature
print(df_father.head())
print('-' * 100)
df_son = df[['childHeight']]    # 이게 label
print(df_son.head())

feature = df_father
label = df_son

feature_f = feature.values.flatten()
label_f = label.values.flatten()
print('부모키 자식키 상관관계 : ', np.corrcoef(feature_f, label_f))   # 상관계수 0.3923835

feature_train, feature_test, label_train, label_test = train_test_split(feature, label,\
                       test_size = 0.3, random_state=1)

model = Sequential()
model.add(Input(shape = (1,)))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
history = model.fit(feature_train, label_train, validation_data=(feature_test, label_test),
                    epochs = 100, batch_size = 1, verbose = 0, shuffle = True)
print(model.summary())

pred = model.predict(feature_test)
print('실제 아들의 키 데이터 : ', label_test.head())
print('예측된 아들의 키 데이터 : ', pred[:5].flatten())

mse = mean_squared_error(label_test,pred)
r2 = r2_score(label_test, pred)
print('테스트 MSE : ', mse)
print('테스트 r2 : ', r2)
