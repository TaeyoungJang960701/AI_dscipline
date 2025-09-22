# 문제5) 이미지 분류
# 1. 라이브러리 임포트 및 데이터셋 설명

# Roboflow Public Datasets 사용
#  - Rock Paper Scissors Classification Dataset : “바위/보/가위” 손 모양 이미지( 약 2,925장 ). 클래스도 3개, 컬러 이미지. 
#    https://public.roboflow.com/classification/rock-paper-scissors?utm_source=chatgpt.com

 

# ** Roboflow에서 Rock Paper Scissors Classification Dataset 데이터 받기 (가장 쉬운 방법: ZIP 다운로드) **
#   -  Roboflow Public 페이지에서 Rock Paper Scissors Classification Dataset의 Train/Valid/Test 압축파일을 내려받는다. 
#   - 아래처럼 풀어줌(예시 경로):
# data/rock-paper-scissors/
#  ├─ train/
#  │   ├─ rock/       *.jpg, *.png ...
#  │   ├─ paper/
#  │   └─ scissors/
#  ├─ valid/
#  │   ├─ rock/
#  │   ├─ paper/
#  │   └─ scissors/
#  └─ test/
#      ├─ rock/
#      ├─ paper/
#      └─ scissors/
# 위와 같이 클래스별 하위 폴더 구조만 맞으면 Keras가 자동으로 라벨을 매겨 준다.
#   - Keras로 불러오기 + feature/label 일부 출력


# 2. 하이퍼파라미터 및 데이터 경로 설정
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)   # 이미지 크기 지정
BATCH    = 32           # 배치 크기 지정

# 학습/검증/테스트 데이터 경로 지정
train_dir = "C:\\Users\\htw02\\Downloads\\Rock Paper Scissors.v1-raw-300x300.folder\\train"
valid_dir = "C:\\Users\\htw02\\Downloads\\Rock Paper Scissors.v1-raw-300x300.folder\\valid"
test_dir  = "C:\\Users\\htw02\\Downloads\\Rock Paper Scissors.v1-raw-300x300.folder\\test"

# 3. 디렉터리에서 이미지 분류용 데이터셋 생성
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",            # 폴더명으로부터 라벨 자동 추출
    label_mode="int",             # [0..C-1] 정수 라벨
    image_size=IMG_SIZE,           # 이미지 크기 통일
    batch_size=BATCH,              # 배치 크기 지정
    shuffle=True,                  # 데이터 섞기
    seed=42,                       # 랜덤 시드 고정
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=42,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False                 # 테스트셋은 셔플하지 않음
)

# 4. 클래스 이름 확인 및 데이터셋 샘플 출력
class_names = train_ds.class_names
print("class_names:", class_names)    # 예: ['paper', 'rock', 'scissors']

# 학습 데이터셋에서 첫 배치의 이미지와 라벨 정보 출력
for images, labels in train_ds.take(1):
    print("features shape:", images.shape)   # (B, H, W, 3)
    print("labels shape:", labels.shape)     # (B,)
    print("labels (first 10):", labels[:10].numpy())
    print("labels mapped (first 10):", [class_names[i] for i in labels[:10].numpy()])

img_generate = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)
# tf.data.Dataset을 넘파이 배열로 변환
x_train = []
y_train = []
for images, labels in train_ds:
    x_train.append(images.numpy())
    y_train.append(labels.numpy())
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# 증강할 이미지 선택
augument_size = 2000
idx = np.random.randint(x_train.shape[0], size=augument_size)
x_src = x_train[idx]
y_src = y_train[idx]

# 증강 이미지 생성
gen = img_generate.flow(x_src, y_src, batch_size=augument_size, shuffle=False)
x_augument, y_augument = next(gen)

# 원본과 증강 데이터 합치기
x_train_aug = np.concatenate((x_train, x_augument), axis=0)
y_train_aug = np.concatenate((y_train, y_augument), axis=0)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(3, activation='softmax')  # 다항분류
])


# EarlyStopping 콜백 추가
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(
    x_train_aug, y_train_aug,
    validation_data=valid_ds,
    epochs=60,
    batch_size=32, # 배치 크기 완화
    verbose=1,
    callbacks=[early_stopping]
)


# 5. 학습/검증 손실 및 정확도 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

# 손실 시각화
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.title('Loss')

# 정확도 시각화
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy')

plt.tight_layout()
plt.show()
plt.close()
