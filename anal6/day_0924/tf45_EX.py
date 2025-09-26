# 문제6) 이미지 분류

# CIFAR-100 dataset 사용    ( 얘 말고 다른 데이터를 이용해도 됨. 음식, 사무용 집기, 라면(우동) ... )
# 특징
#  - 클래스 수: 100개 (예: 사과, 버스, 산, 고래, 시계 등)
#  - 샘플 수: 60,000장, 학습용(train): 50,000장, 테스트용(test): 10,000장
#  - 이미지 크기: 32×32 RGB (작은 해상도)
#  - 레이블 구조: 100개 fine labels (세부 클래스), 20개 coarse labels (상위 클래스 그룹)

# 기본 CNN으로도 학습 가능하지만, 성능을 높이려면
#  - 데이터 증강(ImageDataGenerator / tf.image)
#  - 전이학습(사전학습 모델)
#  - 정규화/드롭아웃/배치정규화 등을 함께 쓰는 게 효과적


# -- 전체 흐름 요약 --

#   작업1 :  CIFAR-100 dataset  분류 모델 작성 (MovileNetV2 모델로 전이학습, 파인튜닝)
#   작업2 : 작성한  분류 모델 사용

#               웹 브라우저에서 이미지 선택
#                    → 장고 웹서버에 저장 → 서버 내부에서 시각화로 확인(matplotlib) + 딥러닝 분류

#                   → 클라이언트에 분류 결과만 반환하기

# 작업2를 좀더 구체적으로 보면
#  1) 클라이언트
#     : index.html에서 파일선택 버튼을 눌러 로컬 컴퓨터의 이미지 파일을 선택하고 화면에 선택된 이미지 출력
#     : 분류결과요청 버튼 클릭 → AJAX 전송 (axios 모듈 사용)
#  2) 서버(Django)
#     : 수신된 이미지 파일 저장 → PIL + Matplotlib(imshow)으로 확인 → 딥러닝 분류 모델로 추론
#     : 응답(JSON): 분류 결과만 반환(예 : bus)
#  3) 클라이언트
#     : 기존 이미지 아래에 이미지 분류 결과 문자열을 화면에 출력

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import numpy as np

ds_info = tfds.builder('food101').info

(train_ds, val_ds, test_ds), ds_info = tfds.load(
    'food101',
    split = ['train[70%:]', 'train[70%:85%]', 'train[:85%]'],
    with_info = True,
    as_supervised = True
)

print('train_ds count:', ds_info.splits['train[70%:]'].num_examples)
print('val_ds count:', ds_info.splits['train[70%:85%]'].num_examples)
print('test_ds count:', ds_info.splits['train[:85%]'].num_examples)
print(ds_info)

total = ds_info.splits['train'].num_examples
print('train 데이터 전체 개수' , total)             # 75750
print('train 갯수 : ', int(total * 0.75))           # 56812
print('validation 갯수 : ', int(total * 0.15))      # 11362
print('test 갯수 : ', int(total * 0.15))            # 11362

# 샘플 크기
for image, label in train_ds.take(1):
    print('원본 1장 : ', image.shape, label.numpy())    # 원본 1장 :  (512, 512, 3) 66

# 레이블 확인
get_label_name = ds_info.features['label'].int2str
print(get_label_name(1))

# 이미지 한장 시각화

for image, label in train_ds.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.axis('off')
    plt.show()
plt.close()

# 전처리
IMG_SIZE = 160

def format_ex(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = train_ds.map(format_ex, num_parallel_calls = tf.data.AUTOTUNE)
validation = val_ds.map(format_ex, num_parallel_calls = tf.data.AUTOTUNE)
test = test_ds.map(format_ex, num_parallel_calls = tf.data.AUTOTUNE)

for img, label in train.take(1):
    print('전처리 결과 type : ', img.dtype)
    print('전처리 결과 shape : ', img.shape)
    print('min/max : ', float(tf.reduce_min(img)), float(tf.reduce_max(img)))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

train_batches = (
    train
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_batches = (
    validation
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_batches = (
    test
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

for image_single, label_single in train_ds.take(2):
    print('원본 단일 이미지 shape:', image_single.numpy().shape)
    print('레이블:', label_single.numpy())
    # 원본 단일 이미지 shape: (512, 512, 3)
    # 레이블: 37 이건 음식 라벨이야

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights = 'imagenet'
)

images_batch, labels_batch = next(iter(train_batches))
feature_batch = base_model(images_batch)

print('입력 배치 shape : ', images_batch.shape)                         # (32, 160, 160, 3)
print('특징맵 배치 shape : ', feature_batch.shape)                      # (32, 5, 5, 1280)

global_avg = tf.keras.layers.GlobalAveragePooling2D()(feature_batch)
print('GAP 이후 shape : ', global_avg.shape)                            # (32, 1280)

# Sequential Api
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = IMG_SHAPE),
    base_model,     # 특징 추출기 - 트레이너블 false 해서 얼릴거야
    tf.keras.layers.GlobalAveragePooling2D(),       # GAP
    tf.keras.layers.Dense(101, activation = 'softmax') # 푸드 101에는 101가지 클래스가 잇으니깐 출력을 101
])
base_model.trainable = False

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_batches, validation_data = validation_batches, epochs = 5, verbose = 1)

test_loss, test_accuracy = model.evaluate(test_batches, verbose = 0)
print(f'test loss : {test_loss:.4f}\ntest accuracy : {test_accuracy:.4f}')


# history 확인
print(history.history.keys())

# 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(loss))

plt.figure(figsize = (12, 5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'train acc')
plt.plot(epochs_range, val_acc, label = 'validation acc')
plt.legend(loc = 'lower right')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'train loss')
plt.plot(epochs_range, val_loss, label = 'validation loss')
plt.legend(loc = 'upper right')

plt.show()
plt.close()

# Fine-tunning
# backbone(MobileNetV2) 동결 해제
base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

check_path_ft = 'finetune_best.keras'

callback_ft = [
    tf.keras.callbacks.ModelCheckpoint(
        check_path_ft,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = True,
        verbose = 1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,
        patience = 2,
        verbose = 1
    ),


    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        restore_best_weights = True,
        patience = 4,
        verbose = 1
    )
]
EPOCHS_TRANSFER = 5
EPOCHS_FINETUNE = 5

history_ft = model.fit(
    train_batches,
    validation_data = validation_batches,
    epochs = EPOCHS_TRANSFER + EPOCHS_FINETUNE,
    initial_epoch = len(history.history['loss']),
    callbacks = callback_ft, verbose = 1
                       )

test_loss, test_acc = model.evaluate(test_batches, verbose = 0)
print(f'fine tune 후의 test loss : {test_loss:.4f}, fine tune 후의 test acc : {test_acc:.4f}')

def concat_hist_func(h1,h2):
    keys = h1.history.keys()
    out = {}
    for k in keys:
        out[k] = h1.history[k] + h2.history[k]
    return out

hist_all = concat_hist_func(history, history_ft)

acc = hist_all['accuracy']
val_acc = hist_all['val_accuracy']

loss = hist_all['loss']
val_loss = hist_all['val_accuracy']

epochs = range(1, len(acc) + 1)
split_epoch = EPOCHS_TRANSFER

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)

plt.plot(epochs, acc, marker = 'o', label = 'train acc')
plt.plot(epochs, val_acc, marker = 's', label = 'val acc')
for i, v in enumerate(acc):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha = 'center', va = 'bottom', fontsize = 8)

for i, v in enumerate(val_acc):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha = 'center', va = 'bottom', fontsize = 8)

plt.axvline(split_epoch, linestyle = '--', alpha = 0.6, label = 'Fine-tunning go')
plt.title('Accuracy (Transfer -> Fine tune)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')

plt.subplot(1,2,2)
plt.plot(epochs, loss, marker = 'o', label = 'train loss')
plt.plot(epochs, val_loss, marker = 's', label = 'val loss')
for i, v in enumerate(loss):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha = 'center', va = 'bottom', fontsize = 8)

plt.axvline(split_epoch, linestyle='--', alpha=0.6, label='Fine-tuning go')
plt.title('Loss (Transfer -> Fine tune)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# model.save('/content/drive/MyDrive/food101.h5')

