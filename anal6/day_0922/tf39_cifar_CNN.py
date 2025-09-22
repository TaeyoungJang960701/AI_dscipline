# cifar-10 dataset으로 이미지 분류 모델을 작성(CNN 사용)
# 총 10개의 label과 6만장의 color 이미지 학습. 32 * 32
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Functional api 모델 정의
def conv_block(x, filters):
    x = Conv2D(filters, 3, padding = 'same', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

inputs = Input(shape = (32,32,3))

# stage1
x = conv_block(inputs, 32)
x = conv_block(x, 32)
x = MaxPooling2D()(x)

# stage2
x = conv_block(x, 64)
x = conv_block(x, 64)
x = MaxPooling2D()(x)

# stage3
x = conv_block(x, 128)
x = conv_block(x, 128)
x = MaxPooling2D()(x)


from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout


x = GlobalAveragePooling2D()(x)

x = Dropout(0.2)(x)
x = Dense(units = 128, activation = 'relu')(x)

x = Dropout(0.2)(x)
outputs = Dense(units = 10, activation = 'softmax')(x)


model = Model(inputs, outputs, name = 'CIFAR10_CNN')
print(model.summary())

model.compile(optimizer = Adam(learning_rate = (1e-3)), 
              loss = 'categorical_crossentropy', metrics = ['accuracy'])
es = EarlyStopping(monitor = 'val_accuracy', patience = 6, restore_best_weights=True)

history = model.fit(x_train, y_train, 
                    epochs = 100,
                    batch_size = 64,
                    validation_split = 0.1,
                    shuffle = True,
                    callbacks = es,
                    verbose = 2)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose = 0)
print(f'test accuracy : {test_accuracy:.5f} | loss : {test_loss:.5f}')