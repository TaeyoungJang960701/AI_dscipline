# fashion mnist dataset으로 CNN 처리
# 실습 1 : Conv + Dense

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0
# print(x_train[0])

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))

for c in range(16):
    plt.subplot(4,4,c + 1)
    plt.imshow(x_train[c].reshape(28,28), cmap = 'gray')

plt.show()

print(y_train[:16])

# 실습 1 : Conv + Dense
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (28,28,1)),
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3)),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3)),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units = 10, activation = 'softmax'),
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc']
              )
print(model.summary())

history = model.fit(x_train, y_train, epochs = 15,validation_split=0.25, verbose = 2)

print(model.evaluate(x_test, y_test, verbose = 0))

plt.figure(figsize = (12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'b-', label = 'accuracy')
plt.plot(history.history['val_acc'], 'r--', label = 'val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()

# 실습 2 : (Conv + Pooling) + Dense

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (28,28,1)),

    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units = 10, activation = 'softmax'),
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc']
              )
print(model.summary())

history = model.fit(x_train, y_train, epochs = 15,validation_split=0.25, verbose = 2)

print(model.evaluate(x_test, y_test, verbose = 0))

plt.figure(figsize = (12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'b-', label = 'accuracy')
plt.plot(history.history['val_acc'], 'r--', label = 'val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()

# 실습 3 : 효율 향상을 위해 성능 좋은 기본 네트워크 일부 도용 (Conv + Pooling) + Dense

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (28,28,1)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc']
              )
print(model.summary())

history = model.fit(x_train, y_train, epochs = 15,validation_split=0.25, verbose = 2)

print(model.evaluate(x_test, y_test, verbose = 0))

plt.figure(figsize = (12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'b-', label = 'accuracy')
plt.plot(history.history['val_acc'], 'r--', label = 'val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()

# 이미지 증강 후 모델 생성
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

img_gener = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.15,
    shear_range = 0.5,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    horizontal_flip = True,
    vertical_flip = False,
)

augment_size = 20000
randidx = np.random.randint(x_train.shape[0], size = augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = img_gener.flow(x_augmented,
                             size = np.zeros(augment_size), batch_size = augment_size,shuffle = False).next()[0]

train_x = np.concatenate((x_train, x_augmented))
train_y = np.concatenate((y_train, y_augmented))

print(train_x.shape)
print(train_y.shape)



model4 = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (28,28,1)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Dense(units=10, activation='softmax')
])

model4.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc']
              )
print(model4.summary())

history4 = model4.fit(x_train, y_train, epochs = 15,validation_split=0.25, verbose = 2)

print(model4.evaluate(x_test, y_test, verbose = 0))

plt.figure(figsize = (12,4))

plt.subplot(1,2,1)
plt.plot(history4.history['loss'], 'b-', label = 'loss')
plt.plot(history4.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history4.history['acc'], 'b-', label = 'accuracy')
plt.plot(history4.history['val_acc'], 'r--', label = 'val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()

