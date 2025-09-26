# MobileNetV2 를 전이학습(미세조정)하여 꽃(클래스 5개) 분류 모델 작성

import os                               # 파일 경로/폴더 작업
import numpy as np                      # 수치 연산 (연산, 행렬)
import matplotlib.pyplot as plt         # 그래프/시각화
import tensorflow_datasets as tfds      # public dataset임
import tensorflow as tf                 # 딥러닝 프레임워크

(train_ds, val_ds), ds_info = tfds.load(
    'tf_flowers',
    split = ['train[:80%]', 'train[80%:]'],
    with_info = True,
    as_supervised = True,        # 반환 타입이 True : tuple, False : dict 타입으로 간다
    shuffle_files = True
)

print(train_ds)
print(val_ds)
# print(raw_test)
# print(metadata)

total = ds_info.splits['train'].num_examples
print('train 원본(전체) 갯수 : ', total)            # 3670
print('raw train 갯수 : ', int(total * 0.8))        # 2936
print('raw validation 갯수 : ', int(total * 0.1))   # 367
print('raw test 갯수 : ', int(total * 0.1))         # 367

for image, label in train_ds.take(1):
    print(type(image), type(label))

# 샘플 크기
for image, label in train_ds.take(1):
    print('원본 1장 : ', image.shape, label.numpy())

# 레이블 확인
get_label_name = ds_info.features['label'].int2str
print(get_label_name(1))

import matplotlib.pyplot as plt

# Take one example before batching and prefetching for displaying
for image, label in train_ds.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.axis('off')
    plt.show()

# 전처리
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds =val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 사전학습된 모델(backbone) 불러오기
base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SIZE + (3,),  # (160, 160) -> (160, 160, 3)
    include_top = False,            # 특징 추출기만 가져오는 옵션
    weights = 'imagenet',
)

base_model.trainable = False        # 이 문구가 전이학습의 핵심

model = tf.keras.Sequential([
    base_model,                                 # MobileNetV2의 Conv와 가중치 역할만 사용
    tf.keras.layers.GlobalAveragePooling2D(),   # 특성맵
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(ds_info.features['label'].num_classes, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

model.fit(train_ds, validation_data = val_ds, epochs = 5)

loss, acc = model.evaluate(val_ds)

print(f'최종분류 정확도 - acc : {acc:.4f}')

# Fine-tunning (전이학습 후 성능을 좀 더 끌어올리고자 할 때 시도하는것)

base_model.trainable = True

print(f'total layers : {len(base_model.layers)}')   # 154 굉장히 많아보이긴 하지만 학습에 핵심적이지 않은게 많대

# for i, layer in enumerate(base_model.layers):
#     if layer.trainable:
#         print(f'["{i:03}] {layer.name}')

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
              loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
              )
# optimizer에다가 'adam'이런식으로 바로 써버리면 러닝레이트를 못매겨줘 그러면 디폴트값을 쓰는데 그건 너무 커
# 그래서 클래스로 써서 러닝레이트를 ㅈㄴ 작게 해주는거야

model.fit(train_ds, validation_data = val_ds, epochs = 3)

loss, acc = model.evaluate(val_ds)
print(f'튜닝 후 최종 분류 정확도 : {acc:.4f}')
print(f'튜닝 후 최종 손실 : {loss:.4f}')

# 예측
for image, label in val_ds.take(1):
    sample_images = image
    sample_labels = label
    break

pred_probs = model.predict(sample_images)
# print('pred_probs : ', pred_probs)

pred_classes = tf.argmax(pred_probs, axis = 1)
print('pred_classes : ',pred_classes)
class_names = ds_info.features['label'].names
print('class_names : ' , class_names)

# 예측 인덱스 vs 실제 인덱스
for i in range(len(sample_images)):
    predicted_index = int(pred_classes[i])
    actual_index = int(sample_labels[i])
    predicted_name = class_names[predicted_index]
    actual_name = class_names[acual_index]

    print(f'[{i:02}] Predicted : {predicted_index}({predicted_name})   Actual : {actual_index}({actual_name})')


import matplotlib.pyplot as plt

plt.figure(figsize = (12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i])
    predicted_label = class_names[pred_classes[i]]
    actual_label = class_names[sample_labels[i]]
    color = 'blue' if predicted_label == actual_label else 'red'
    plt.title(f'Predicted : {predicted_label}\nActual : {actual_label}', color = color, fontsize = 10)
    plt.axis('off')
plt.tight_layout()
plt.show()