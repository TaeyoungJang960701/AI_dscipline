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

# import tensorflow as tf
# import numpy as np
# IMG_SIZE = (224, 224)   # 필요에 따라 (128,128) 등으로 변경
# BATCH    = 32

# train_dir = "data/rock-paper-scissors/train"
# valid_dir = "data/rock-paper-scissors/valid"
# test_dir  = "data/rock-paper-scissors/test"

# # 디렉터리에서 이미지 분류용 데이터셋 만들기
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     train_dir,
#     labels="inferred",
#     label_mode="int",             # [0..C-1] 정수 라벨
#     image_size=IMG_SIZE,
#     batch_size=BATCH,
#     shuffle=True,
#     seed=42,
# )

# valid_ds = tf.keras.utils.image_dataset_from_directory(
#     valid_dir,
#     labels="inferred",
#     label_mode="int",
#     image_size=IMG_SIZE,
#     batch_size=BATCH,
#     shuffle=True,
#     seed=42,
# )

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     test_dir,
#     labels="inferred",
#     label_mode="int",
#     image_size=IMG_SIZE,
#     batch_size=BATCH,
#     shuffle=False
# )
# # 클래스 이름 확인
# class_names = train_ds.class_names
# print("class_names:", class_names)    # 예: ['paper', 'rock', 'scissors']
# # --- feature와 label 일부 출력 ---
# for images, labels in train_ds.take(1):
#     print("features shape:", images.shape)   # (B, H, W, 3)
#     print("labels shape:", labels.shape)     # (B,)
#     print("labels (first 10):", labels[:10].numpy())
#     print("labels mapped (first 10):", [class_names[i] for i in labels[:10].numpy()])
# 마지막에 이미지 증강 전,후의 모델 성능 비교 - ROC curve 사용

# 새로운 이미지에 대한 분류 결과 확인

import cv2,os,numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

import tensorflow as tf
import numpy as np
import os

IMG_SIZE = (224, 224)   # 필요에 따라 (128,128) 등으로 변경
BATCH    = 16 # Reduced batch size to save memory

train_dir = '/content/drive/MyDrive/rock_paper_scissor/Rock_Paper_Scissors/train'
test_dir = '/content/drive/MyDrive/rock_paper_scissor/Rock_Paper_Scissors/test'
valid_dir = '/content/drive/MyDrive/rock_paper_scissor/Rock_Paper_Scissors/valid'

if not os.path.exists(train_dir):
    print(f"Error: Training directory not found at {train_dir}")
else:
    # 디렉터리에서 이미지 분류용 데이터셋 만들기
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",             # [0..C-1] 정수 라벨
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=42,
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
        shuffle=False
    )

    # 클래스 이름 확인
    class_names = train_ds.class_names
    print("class_names:", class_names)
    # --- feature와 label 일부 출력 ---
    for images, labels in train_ds.take(1):
        print("features shape:", images.shape)   # (B, H, W, 3)
        print("labels shape:", labels.shape)     # (B,)
        print("labels (first 10):", labels[:10].numpy())
        print("labels mapped (first 10):", [class_names[i] for i in labels[:10].numpy()])

import matplotlib.pyplot as plt

# train_ds에서 첫 번째 배치 가져오기
train_images, train_labels = next(iter(train_ds))
test_images, test_labels = next(iter(test_ds))
valid_images, valid_labels = next(iter(valid_ds))

# 배치의 첫 번째 이미지와 라벨 선택
first_image = train_images[0].numpy().astype("uint8")
first_label = labels[0].numpy()

# 이미지 플롯
plt.imshow(first_image)
plt.title(f"Label: {class_names[first_label]}")
plt.axis("off")
plt.show()

print(train_images.shape)

# train, test 스플릿은 위에서 이미 다 돼잇고
# 이젠 모델 설정
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Input
from tensorflow.keras.models import Sequential

model = Sequential([
    Input(shape = (224, 224,3)),
    Conv2D(112, (3, 3), activation = 'relu'),
    MaxPooling2D((2,2)), # Changed pool size
    Conv2D(224, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)), # Changed pool size
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.3),
    Dense(len(class_names), activation = 'softmax'), # Adjust output layer for 3 classes
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 모델 체크포인트 설정: 학습 중 가장 성능이 좋은 모델 저장
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5',  # 모델 저장 경로
    monitor='val_loss',        # 검증 손실을 모니터링
    save_best_only=True,       # 가장 좋은 성능을 보인 모델만 저장
    mode='min',                # val_loss가 최소일 때 저장
    verbose=1                  # 저장될 때 메시지 출력
)


# Early Stopping 설정: 검증 손실이 개선되지 않으면 학습 조기 중단
early_stopping_callback = EarlyStopping(
    monitor='val_loss',        # 검증 손실을 모니터링
    patience=5,                # 개선되지 않아도 기다릴 epoch 수
    mode='min',                # val_loss가 최소일 때 중단
    verbose=1                  # 중단될 때 메시지 출력
)

# 모델 학습
history = model.fit(
    train_ds,
    epochs=20,  # 학습 epoch 수 설정 (필요에 따라 조정)
    validation_data=valid_ds,
    callbacks=[checkpoint_callback, early_stopping_callback],
    verbose=2
)

# 모델 예측
pred = model.predict(test_images)
print('예측값 : ', (pred >= 0.5).astype(int).reshape(-1))
print('실제값 : ', test_images)

plt.plot(history.history['acc'], label = ['train_accuracy'])
plt.plot(history.history['val_acc'], color = 'red', label = ['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
# plt.tightlayout()
plt.show()
plt.close()

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# 저장된 최적의 모델 불러오기
try:
    best_model = load_model('best_model.h5')
    print("Best model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best_model.h5' exists after training.")
    best_model = None # Set best_model to None if loading fails

if best_model:
    # test_ds에서 이미지와 실제 라벨 가져오기
    test_images, test_labels = next(iter(test_ds))

    # 모델 예측
    predictions = best_model.predict(test_images)

    # 예측된 클래스 인덱스 가져오기
    predicted_classes = np.argmax(predictions, axis=1)

    # 실제 라벨과 예측 라벨 출력 (첫 10개)
    print("Actual labels (first 10):", test_labels[:10].numpy())
    print("Predicted labels (first 10):", predicted_classes[:10])
    print("Actual labels mapped (first 10):", [class_names[i] for i in test_labels[:10].numpy()])
    print("Predicted labels mapped (first 10):", [class_names[i] for i in predicted_classes[:10]])


    # 몇 개의 샘플 이미지와 예측 결과 시각화
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(test_images))): # Display up to 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i].numpy().astype("uint8"))
        plt.title(f"Actual: {class_names[test_labels[i].numpy()]}\nPredicted: {class_names[predicted_classes[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

else:
    print("Cannot perform classification as the model was not loaded.")

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os # os 모듈 추가

try:
    model = load_model('best_model.h5')
    print("Best model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best_model.h5' exists after training.")
    model = None

if model:
    # 판별할 이미지 파일 경로 설정
    # TODO: 이곳에 사용자의 이미지 파일 경로를 입력하세요.
    # 예시: '/content/drive/MyDrive/my_images/my_rock_photo.jpg'
    rock_img_path = '/content/drive/MyDrive/class_images/rock.jpg' # 예시 경로 (실제 이미지 파일 경로로 변경해야 합니다)
    paper_img_path = '/content/drive/MyDrive/class_images/paper.jpg' # 예시 경로 (실제 이미지 파일 경로로 변경해야 합니다)
    scissor_img_path = '/content/drive/MyDrive/class_images/scissor.jpg' # 예시 경로 (실제 이미지 파일 경로로 변경해야 합니다)

    image_paths = {
        'rock': rock_img_path,
        'paper': paper_img_path,
        'scissors': scissor_img_path
    }

    for class_name, img_path in image_paths.items():
        if not os.path.exists(img_path):
            print(f"Error: Image file not found at {img_path}")
            continue # 다음 이미지로 건너뛰기

        # 이미지 로드 및 전처리
        try:
            img = image.load_img(img_path, target_size=IMG_SIZE) # IMG_SIZE는 이전에 정의된 이미지 크기
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

            # 모델 예측
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index] # class_names는 이전에 정의된 클래스 이름 리스트

            # 예측 결과 출력
            print(f"The predicted class for the image '{os.path.basename(img_path)}' is: {predicted_class_name}")

            # 이미지와 예측 결과 시각화 (선택 사항)
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_class_name}")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"An error occurred during processing {img_path}: {e}")

else:
    print("Cannot perform classification as the model was not loaded.")

