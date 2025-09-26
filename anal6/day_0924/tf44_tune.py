# MobileNetV2 를 전이학습하여 개, 고양이 분류 모델 생성
import os                               # 파일 경로/폴더 작업
import numpy as np                      # 수치 연산 (연산, 행렬)
import matplotlib.pyplot as plt         # 그래프/시각화
import tensorflow_datasets as tfds      # public dataset임
import tensorflow as tf                 # 딥러닝 프레임워크

tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info = True,
    as_supervised = True        # 반환 타입이 True : tuple, False : dict 타입으로 간다
)

print(raw_train)
print(raw_validation)
print(raw_test)
print(metadata)

total = metadata.splits['train'].num_examples
print('train 원본(전체) 갯수 : ', total)            # 23262
print('raw train 갯수 : ', int(total * 0.8))        # 18609
print('raw validation 갯수 : ', int(total * 0.1))   # 2326
print('raw test 갯수 : ', int(total * 0.1))         # 2326

# 샘플 크기
for image, label in raw_train.take(1):
    print('원본 1장 : ', image.shape, label.numpy())    # 원본 1장 :  (262, 350, 3) 1

# 레이블 확인
get_label_name = metadata.features['label'].int2str
print(get_label_name(1))    # dog


# 이미지 한장 시각화
import matplotlib.pyplot as plt

for image, label in raw_train.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.axis('off')
    plt.show()


# 전처리
IMG_SIZE = 160

def format_ex(image, label):
    # 1) dtype 변환: TFDS는 보통 uint8 → 연산/정규화용 float32로 캐스팅
    image = tf.cast(image, tf.float32)
    # 2) 스케일 정규화: 0~255 → [-1, 1]
    #    MobileNetV2 권장 입력 범위와 동일(= keras.applications.mobilenet_v2.preprocess_input)
    image = (image / 127.5) - 1.0
    # 3) 해상도 통일: 다양한 원본 크기 → (IMG_SIZE, IMG_SIZE)로 리사이즈
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# AUTOTUNE : CPU 코어개수/리소스 상황에 맞게 자동으로 최적화
# GPU idle time을 최소화
# 각 분할에 전처리 함수 적용(map)
train = raw_train.map(format_ex,num_parallel_calls=tf.data.AUTOTUNE)
validation = raw_validation.map(format_ex,num_parallel_calls=tf.data.AUTOTUNE)
test = raw_test.map(format_ex,num_parallel_calls=tf.data.AUTOTUNE)

# 전처리 결과 확인: 샘플 1장만 꺼내 dtype/shape/값 범위 출력
for img, label in train.take(1):
    print('전처리 결과 type : ',  img.dtype)   # 기대: tf.float32
    print('전처리 결과 shape : ', img.shape)   # 기대: (160, 160, 3)  (IMG_SIZE=160 기준)
    # reduce_min/max는 스칼라 텐서 → float(...)로 파이썬 float로 변환(또는 .numpy() 사용)
    print('min/max : ', float(tf.reduce_min(img)), float(tf.reduce_max(img)))  # 기대: 대략 -1.0 ~ 1.0

# 배치 파이프라인 작성(학습용/검증용)
# 1000의 샘플을 메모리에 가져와 무작위로 섞음 -> 그 다음 데이터 비퍼에 읽어 또 섞
# 미니배치 크기: 한 번에 모델에 공급할 샘플 수 (메모리/속도/일반화 성능의 트레이드오프)
BATCH_SIZE = 32
# 셔플 버퍼 크기: 이만큼의 샘플을 메모리에 담아 무작위로 섞음
# 값이 클수록 섞임이 잘 되지만 메모리 사용량↑. 보통 BATCH_SIZE의 10~50배 권장.
SHUFFLE_BUFFER_SIZE = 1000

# train만 섞어줌
# 학습용 파이프라인: 섞기 → 배치 묶기 → prefetch(입출력/연산 겹치기)
train_batches = (
    train
    .shuffle(SHUFFLE_BUFFER_SIZE)     # 메모리에 버퍼 크기만큼 담아 무작위 셔플
    .batch(BATCH_SIZE)                 # BATCH_SIZE 단위로 묶음
    .prefetch(tf.data.AUTOTUNE)        # CPU 전처리와 GPU 학습을 파이프라이닝(속도 ↑)
)

# 검증/테스트: 셔플 없이 순서 유지(재현성), 배치/프리페치만 사용
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

# 원본(raw_train)에서 전처리/배치 이전의 샘플 2개만 직접 확인
for image_single, label_single in raw_train.take(2):
    # .numpy(): eager 모드에서 텐서를 넘파이로 변환해 출력(그래프 모드면 tf.print 사용)
    print('원본 단일 이미지 shape:', image_single.numpy().shape)  # 예) (H, W, 3) — 원본 해상도라 제각각
    print('레이블:', label_single.numpy())                       # 정수 라벨 (예: 0=cat, 1=dog)

    # base model
# 전이학습용 백본(Feature Extractor)으로 MobileNetV2 로드
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  # 예: (160, 160, 3) — H, W, C(RGB)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,    # 입력 이미지 크기 지정
    include_top=False,        # 최상단 분류기(FC + Softmax) 제거 → 특징 추출기만 사용
    weights='imagenet'        # ImageNet으로 학습된 사전 가중치 로드(전이학습)
)

# 모델 구조 요약 출력(레이어, 출력 텐서 크기, 파라미터 수 확인)
# print(base_model.summary())

# 전처리/batch된 텐서를 통과시켜 특징 맵 얻기
images_batch,labels_batch=next(iter(train_batches))
feature_batch=base_model(images_batch)
print('입력 배치 shape : ',images_batch.shape)     # 예) 입력 배치 shape :    (32, 160, 160, 3)
print('특징맵 배치 shape : ',feature_batch.shape)  # 예) 특징맵 배치 shape :  (32, 5, 5, 1280)

global_avg=tf.keras.layers.GlobalAveragePooling2D()(feature_batch)
print('GAP 이후 shape : ',global_avg.shape)  # 예) GAP 이후 shape :  (32, 1280)

# model 정의
# Sequential api
# model=tf.keras.layers.Sequential([
#     tf.keras.layers.Input(shape=IMG_SHAPE),
#     base_model,   # 특징 추출기(컨볼루션) - 동결상태
#     tf.keras.layers.GlobalAveragePooling2D(),   # GAP
#     tf.keras.layers.Dense(1,activation='sigmoid')
# ])

# Fuctional api
inputs=tf.keras.Input(shape=IMG_SHAPE)
x=base_model(inputs,training=False)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
outputs=tf.keras.layers.Dense(1,activation='sigmoid')(x)
model=tf.keras.Model(inputs,outputs)

base_model.trainable=False

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(train_batches,validation_data=validation_batches,epochs=5,verbose=1)

test_loss,test_acc=model.evaluate(test_batches,verbose=0)
print(f'test loss : {test_loss:.4f}, test acc : {test_acc:.4f}')

# history확인
print(history.history.keys())

# 시각화
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range=range(len(acc))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='train acc')
plt.plot(epochs_range,val_acc,label='validation acc')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='train loss')
plt.plot(epochs_range,val_loss,label='validation loss')
plt.legend(loc='lower right')

plt.show()
plt.close()

# ----- Fine-Tuning(미세조정)-----

# 백본(MobileNetV2) 동결 해제: 이제 일부 레이어를 학습 가능하게 만들 준비
base_model.trainable = True   # 특징추출기 Unfreeze

# 하위 레이어는 유지, 상위 일부만 풀기
fine_tune_at = 100            # 백본의 0~99번 레이어는 동결 유지(= 학습 X)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False   # 하위층 동결 → 과적합/불안정 학습 방지

# trainable 변경 후에는 반드시 재-컴파일!
#    - 학습률은 미세조정 단계에서 더 작게(예: 1e-4 ~ 1e-5)
#    - 손실: 이진 분류면 'binary_crossentropy' (출력층: Dense(1, sigmoid) 가정)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
chk_path_ft = 'finetune_best.keras'
callback_ft = [
    # 최고 성능(val_accuracy 기준) 모델만 저장
    tf.keras.callbacks.ModelCheckpoint(
        chk_path_ft,
        monitor='val_accuracy',   # 검증 정확도를 모니터링
        mode='max',               # 값이 클수록 좋음
        save_best_only=True,      # 최고치 갱신 때만 저장
        verbose=1
    ),
    # 검증 손실이 개선되지 않으면 학습률을 factor 배로 감소(Plateau 대응)
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',       # 손실 기준으로 감지(과적합 시 val_loss가 민감)
        factor=0.5,               # LR을 절반으로 감소
        patience=2,               # 2 epoch 개선 없으면 감소
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True, # Corrected parameter name
        patience=4,
        verbose=1
    )
]
EPOCHS_TRANSFER=5
EPOCHS_FINETUNE=5

# 전이학습이 끝난 후 이어서 학습시작
history_ft=model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=EPOCHS_TRANSFER+EPOCHS_FINETUNE,
    initial_epoch=len(history.history['loss']),
    callbacks=callback_ft,verbose=2
)

test_loss,test_acc=model.evaluate(test_batches,verbose=0)
print(f'fine tune 후에 test loss : {test_loss:.4f}, fine tune 후에 test acc : {test_acc:.4f}')

# 전이학습 vs 미세조정 학습 곡선 결합한 시각화
def concat_hist_func(h1,h2):
  keys=h1.history.keys()
  out={}
  for k in keys:
    out[k]=h1.history[k]+h2.history[k]
  return out

hist_all=concat_hist_func(history,history_ft)
acc=hist_all['accuracy']
val_acc=hist_all['val_accuracy']
loss=hist_all['loss']
val_loss=hist_all['val_loss']

epochs=range(1,len(acc)+1)
split_epoch=EPOCHS_TRANSFER   # 전이학습과 미세조정 경계선 위치

plt.figure(figsize=(12,5))
# 정확도-----
plt.subplot(1,2,1)
plt.plot(epochs,acc,marker='o',label='train acc')
plt.plot(epochs,val_acc,marker='s',label='val acc')
for i,v in enumerate(acc):
     plt.text(epochs[i],v,f'{v*100:.1f}%',ha='center',va='bottom',fontsize=8)

for i,v in enumerate(val_acc):
     plt.text(epochs[i],v,f'{v*100:.1f}%',ha='center',va='bottom',fontsize=8)

plt.axvline(split_epoch,linestyle='--',alpha=0.6,label='Fine-tuning go')
plt.title('Accuracy (Transfer -> Fine tune)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
# 손실-----
plt.subplot(1,2,2)
plt.plot(epochs,loss,marker='o',label='train loss')
plt.plot(epochs,val_loss,marker='s',label='val loss')
for i,v in enumerate(loss):
     plt.text(epochs[i],v,f'{v*100:.1f}%',ha='center',va='bottom',fontsize=8)

for i,v in enumerate(val_loss):
     plt.text(epochs[i],v,f'{v*100:.1f}%',ha='center',va='bottom',fontsize=8)

plt.axvline(split_epoch,linestyle='--',alpha=0.6,label='Fine-tuning go')
plt.title('Loss (Transfer -> Fine tune)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

