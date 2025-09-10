import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [1]])          # XOR

model = Sequential()
model.add(Input(shape = (2,)))
model.add(Dense(units = 5, activation = 'relu'))
model.add(Dense(units = 5, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

print(model.summary())


model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.01),metrics = ['accuracy'])
history = model.fit(x, y, epochs = 100, batch_size = 1, verbose = 0)
loss_metrics = model.evaluate(x,y)
print('loss_metrics')


pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과 : ', pred.ravel())

print(model.weights)

print(history.history['loss'][:10])
print(history.history['accuracy'][:10])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.xlabel('epochs')
plt.legend(loc = 'best')
plt.show()

# 여기서 실행되는 곱셈에 대한 식은
# tf.multiply((x1,w1) + b) 뭐 이런식이 아니라
# tf.matmul(x1,w1) 뭐 이런식이야 왜 맷멀이냐? 이건 매트릭스이기 때문에 그래 처음에 준 x값이 매트릭스엿기 때문에
