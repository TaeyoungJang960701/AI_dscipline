# mnist tensorflow as tf
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 구조 변경(차원)
print(x_train.shape)        # (60000, 28, 28)
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_train.shape)        # (60000, 28, 28, 1)

# 모델 정의
inputs = tf.keras.layers.Input(shape = (28, 28, 1))

"""
방법 1.

x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = 'same', activation = 'relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
x = tf.keras.layers.Dropout(rate = 0.2)(x)

x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

# Fully Connected Layers
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units = 64, activation = 'relu')(x)
x = tf.keras.layers.Dropout(rate = 0.3)(x)
x = tf.keras.layers.Dense(units = 32, activation = 'relu')(x)
x = tf.keras.layers.Dropout(rate = 0.2)(x)

outputs = tf.keras.layers.Dense(units = 10, activation = 'softmax')(x)
"""

# 방법 2. - BatchNormalization : Conv / Dense 뒤에 배치 - 학습을 안정화시키고, 수렴 가속화 그래서 많이 쓴다
# use_bias = False  : Conv / Dense 의 bias 제거
x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = 'same', use_bias = False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Dropout(rate = 0.25)(x)

x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', use_bias = False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Dropout(rate = 0.25)(x)

x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', use_bias = False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Dropout(rate = 0.25)(x)


# Fully Connected Layers
x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(64, use_bias = False)(x) # Pass input tensor to the Dense layer
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(rate = 0.3)(x)

x = tf.keras.layers.Dense(32)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(rate = 0.3)(x)

outputs = tf.keras.layers.Dense(10, activation = 'softmax')(x) # Pass input tensor to the Dense layer

# 모델 객체 생성
model = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = 'mnist_cnn_func')
print(model.summary())

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

es = tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs = 100, batch_size = 128, validation_split = 0.1,
                    callbacks = [es], verbose = 2
                    )

# 모델 평가
train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0)
print(f'train_loss : {train_loss:.4f} train_accuracy : {train_acc:.4f}')
print(f'test_loss : {test_loss:.4f} test_accuracy : {test_acc:.4f}')