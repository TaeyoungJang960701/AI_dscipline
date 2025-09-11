import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
import numpy as np

x_data = np.array([1.,2.,3.,4.,5.]).reshape(-1, 1)
y_data = np.array([1.2, 2.0, 3.0,3.5,5.3]).reshape(-1, 1)
print(x_data)

# print('상관계수 : ', np.corrcoef(x_data, y_data()))
print('상관계수 : ', np.corrcoef(x_data.ravel(), y_data.ravel()))
# 상관계수를 보여주는 클래스는 1차원 데이터만 받아
# 근데 처음 데이터 선언을 2차원으로 해버려서 ravel()클래스로 차원을 낮춰서 쓴거야

model = Sequential()
model.add(Input(shape = (1,)))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))
print(model.summary())

model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['mse'])
model.fit(x_data, y_data, batch_size = 1, epochs = 10, verbose = 1, shuffle = True)
print(model.evaluate(x_data,y_data))

pred = model.predict(x_data)
print('pred : ', pred.ravel())
print('real : ', y_data.ravel())

# 결정계수
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, pred))

import matplotlib.pyplot as plt
plt.scatter(x_data, y_data, color = 'r', marker = 'o', label = 'real')
plt.plot(x_data, pred, 'b--', label = 'pred')
plt.show()
plt.close()