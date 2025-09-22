# CNN + ImageDataGenerator : fashion_MNIST 이미지 보강

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255
x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255
print(x_train[:2])
print(y_train[:2])

y_train = to_categorical(y_train)
# print(y_train)
y_test = to_categorical(y_test)

# 이미지 시각화
plt.figure(figsize = (10, 10))
for c in range(0, 99):
    plt.subplot(10,10, c + 1)
    plt.axis('off')
    plt.imshow(x_train[c].reshape(28, 28), cmap = 'gray')
plt.show()

print(x_train.shape)    # 원본 자료의 차원 (60000, 28, 28, 1)
print(y_train.shape)    # (60000, 10)

# 이미지 보강
"""
img_generate = ImageDataGenerator(
    rotation_range = 10,        # 랜덤하게 그림 회전 (각도 0 ~ 180도 회전)
    zoom_range = 0.1,           # 확대, 축소
    shear_range = 0.5,          # 축을 중심으로 전환(모양 기울이기)
    width_shift_range = 0.1,    # 수평 이동
    height_shift_range = 0.1,   # 수직 이동
    horizontal_flip = True,     # 좌우 수평 전환
    vertical_flip = False       # 상하 수직 전환
)

augment_size = 100              # 증강 샘플 수 100
idx = np.random.randint(x_train.shape[0], size = augment_size)
# print(idx)

x_src = x_train[idx].copy()
y_src = y_train[idx].copy()
# print(x_src)
# print(y_src)

gen = img_generate.flow(            # flow_from_directory
    x_src,
    y = np.zeros(augment_size),
    batch_size = augment_size,
    shuffle = False,
    seed = 42,                      # 증강 재현성 고정(난수 고정)
)

x_augmented = next(gen)[0]          # flow 반환값은 generator 객체임. 그래서 next()로 다음 배치를 꺼내옴
# 필요하며 원본에 합치기
x_train_aug = np.concatenate([x_train, x_augmented], axis = 0)
y_train_aug = np.concatenate([y_train, y_src], axis = 0)
print(x_train_aug.shape)

# 확인용 시각화
n = 16
fig, axes = plt.subplots(1, n, figsize = (n, 4))
for i, ax in enumerate(axes):
    ax.imshow(x_augmented[i].squeeze(), cmap = 'gray')
    ax.axis('off')


plt.tight_layout()
plt.show()
"""

img_generate = ImageDataGenerator(
    rotation_range = 10,        # 랜덤하게 그림 회전 (각도 0 ~ 180도 회전)
    zoom_range = 0.1,           # 확대, 축소
    shear_range = 0.5,          # 축을 중심으로 전환(모양 기울이기)
    width_shift_range = 0.1,    # 수평 이동
    height_shift_range = 0.1,   # 수직 이동
    horizontal_flip = True,     # 좌우 수평 전환
    vertical_flip = False       # 상하 수직 전환
)

augment_size = 30000

rand_idx = np.random.randint(x_train.shape[0], size = augment_size)
# print(idx)

x_augment = x_train[rand_idx].copy()
y_augment = y_train[rand_idx].copy()
# print(x_src)
# print(y_src)

gen = img_generate.flow(            # flow_from_directory
    x_augment,
    y_augment,
    batch_size = augment_size,
    shuffle = False,
    seed = 42,                      # 증강 재현성 고정(난수 고정)
)

x_augment, y_augment = next(gen)

# 원본에 합치기
x_train = np.concatenate([x_train, x_augment], axis = 0)
y_train = np.concatenate([y_train, y_augment], axis = 0)
print(x_train.shape)        # (90000, 28, 28, 1)
print(y_train.shape)        # (90000, 10)

# CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (28, 28, 1)),

    # 데이터 특징 추출 레이어
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding ='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.1),

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding ='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.1),

    tf.keras.layers.Flatten(),

    # 분류기 레이어
    tf.keras.layers.Dense(units = 128, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.3),

    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.3),

    tf.keras.layers.Dense(units = 10, activation = 'softmax'),
    tf.keras.layers.Dropout(rate = 0.3),

])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

#

# 모델 최적화 설정
MODEL_DIR = './mnist/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = './mnist/{epoch:02d}-{val_loss:.2f}.keras'
chkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only=True, verbose = 2)

earlystop = EarlyStopping(monitor = 'val_loss', patience = 5)

history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs = 100,
                    batch_size = 64,
                    verbose = 2,
                    callbacks =[earlystop, chkpoint])
print(['Test Accuracy : %.4f'%(model.evaluate(x_test, y_test)[1])])

# 시각화
plt.figure(figsize = (12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], marker = 'o', color = 'red', label = 'acc')
plt.plot(history.history['val_accuracy'], marker = 's', color = 'blue', label = 'val_acc')
plt.xlabel('epochs')
plt.ylim(0.05, 1)
plt.ylabel('accuracy')
plt.legend(loc = 'lower left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], marker = 'o', color = 'red', label = 'loss')
plt.plot(history.history['val_loss'], marker = 's', color = 'blue', label = 'val_loss')
plt.xlabel('epochs')
plt.ylim(0.05, 1)
plt.ylabel('loss')
plt.legend(loc = 'upper right')

plt.show()