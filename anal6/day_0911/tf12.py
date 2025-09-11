# 다중선형회귀모델 + 텐서보드

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import shutil, os, datetime as dt

# 작년의 5명의 3회 모의고사 점수로 학습 후 본 고사 점수 예측
# 새 데이터로 점수 예측
tf.config.run_functions_eagerly(True)


x_data = np.array([[70, 85, 80],[71, 89, 78],[50, 80, 60],[66, 30, 60],[50, 25, 10]])
y_data = np.array([[73,82,72, 57, 34]]).T

print('1) Sequential api ------')
model = Sequential()
model.add(Input(shape = ((3,))))
# model.add(Dense(units = 1, activation = 'linear'))    # 레이어 한개만 쓸거야
model.add(Dense(units = 8, activation = 'relu', name = 'a'))
model.add(Dense(units = 4, activation = 'relu', name = 'b'))
model.add(Dense(units = 1, activation = 'linear', name = 'c'))

print(model.summary())

opti = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer = opti, loss = 'mse', metrics = ['mse'])
history = model.fit(x_data, y_data, batch_size = 1, epochs = 50, verbose = 0)

# 시각화
# plt.plot(history.history['loss'])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()
# plt.close()
loss_metrics = model.evaluate(x = x_data, y = y_data)
print('loss_metrics : ', loss_metrics)
print('설명력 : ',r2_score(y_data,model.predict(x_data)))

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

print('2) functional api -----')
inputs = Input(shape = (3,))
h1 = Dense(units = 8, activation = 'relu', name = 'a')(inputs)
h2 = Dense(units = 4, activation = 'relu', name = 'b')(h1)
outputs = Dense(units = 1, activation = 'linear', name = 'c')(h2)

model = Model(inputs, outputs, name = 'linear_model')

# TensorBoard -----------------
BASE = 'logs'     # 기본 로그 저장 디렉토리명
shutil.rmtree(BASE, ignore_errors = True)   # 해당 디렉토리 삭제
RUN = os.path.join(BASE, 'test')
os.makedirs(RUN, exist_ok = True)

tb = TensorBoard(log_dir=RUN, histogram_freq = 1, write_graph = True)   # 매 epoch마다 histogram 만들어라
# --------------------------------

# Create a new optimizer instance for this model
opti_functional = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer = opti_functional, loss = 'mse', metrics = ['mse'])
model.fit(x_data, y_data, batch_size = 1, epochs = 50, verbose = 2, callbacks = [tb])

%load_ext tensorboard
%tensorboard --logdir logs

from tensorflow.keras.utils import plot_model
plot_model(model, to_file = 'model.png', show_shapes = True, show_layer_names = True)