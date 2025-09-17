# MNIST dataset(숫자 이미지)으로 숫자 분류 모델
# 숫자 손글씨 이미지에 대한 데이터와 라벨이 포함되어 있으며
# 6만개의 트레이닝 데이터와 1만개의 테스트 데이터가 있다

import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0])   # 0번째 feature
print(y_train[0])   # 0번째 label

# for i in x_train[0]:
#     for j in i:
#         sys.stdout.write('%s  '%j)
#     sys.stdout.write('\n')

plt.imshow(x_train[0], cmap = 'gray')
plt.show()

x_train = x_train.reshape(60000,784).astype('float32')      # 28 by 28 -> 784열 변경
x_test = x_test.reshape(10000,784).astype('float32')      # 28 by 28 -> 784열 변경
# print(x_train[0])

x_train /= 255.0    # 정규화 : 필수는 아니다 하지만 해주면 모델 성능 향상됨
x_test /= 255.0
# print(x_train[0])

# label : OneHot encoding - 출력층 활성화 함수를 softmax
print(set(y_train))
# print('-' * 100)
print(y_train[0])

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(y_train[0])

# validation data
x_val = x_train[50000:60000]
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[0:50000] # Corrected slicing for y_train

print(x_val.shape, x_train.shape)       # (10000, 784) (50000, 784)

# model
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization

model = Sequential()

model.add(Input(shape = (784,)))
# model.add(Dense(units = 64))
# model.add(Activation = 'relu')

# model.add(Dropout(rate = 0.2))
# model.add(Dense(units = 32))
# model.add(Activation = 'relu')

# model.add(Dropout(rate = 0.2))
# model.add(Dense(units = 10))
# model.add(Activation = 'softmax')
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, epochs = 10,
                    batch_size = 128, validation_data = (x_val, y_val), verbose = 2)


print('loss : ', history.history['loss'])
print('val_loss : ', history.history['val_loss'])
print('accuracy : ', history.history['accuracy'])
print('val_accuracy : ', history.history['val_accuracy'])

epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], label = 'loss')
plt.plot(epochs, history.history['val_loss'], label = 'val_loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], label = 'accuracy')
plt.plot(epochs, history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, batch_size = 128, verbose = 0)
print('loss : ', score[0])
print('accuracy : ', score[1])

# model.save('tf27_model.keras')
save_path = '/content/drive/MyDrive/mysou/tf27_model.keras'
model.save(save_path)

# del model

# mymodel = tf.keras.models.load_model('tf27_model.keras')
mymodel = tf.keras.models.load_model(save_path)

# print(x_test[:1], x_test[:1].shape)
plt.imshow(x_test[:1].reshape(28,28), cmap = 'Greys')
plt.show()

pred = mymodel.predict(x_test[:1])
print('pred : ', pred)
print('예측값 : ', np.argmax(pred, 1))
print('실제값 : ', y_test[:1])
print('실제값 : ', np.argmax(y_test[:1], 1))
