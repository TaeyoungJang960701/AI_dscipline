# CNN : 개 / 고양이 이미지 분류
import os, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(1)
tf.random.set_seed(1)

data_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_path = tf.keras.utils.get_file(
    fname = 'cats_and_dogs_filtered.zip',
    origin = data_url,
    extract = False,
    cache_dir = '.',
    cache_subdir = '',
)
print(zip_path)

if not os.path.exists('./cats_and_dogs_filtered'):
    with zipfile.ZipFile(zip_path, 'r') as obj:
        obj.extractall('.')
        print('Extract ok zzz')

# 경로 확인
PATH = './cats_and_dogs_filtered'                       # root directory
train_dir = os.path.join(PATH, 'train')                 # 학습폴더명
validation_dir = os.path.join(PATH, 'validation')       # 검증폴더명

BATCH_SIZE = 128
EPOCHS = 15
IMG_HEIGHT, IMG_WIDTH = 150, 150

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
val_cats_dir = os.path.join(validation_dir, 'cats')
val_dogs_dir = os.path.join(validation_dir, 'dogs')

for p in [train_dir, train_cats_dir, train_dogs_dir, validation_dir, val_cats_dir, val_dogs_dir]:
    print(p, '->', os.path.exists(p))

# 개수 파악
print('cats(train) : ', len(os.listdir(train_cats_dir)),        # cats(train) :  1000 , dogs(train) :  1000
      ',', 'dogs(train) : ', len(os.listdir(train_dogs_dir)))
print('cats(validation) : ', len(os.listdir(val_cats_dir)),     # cats(validation) :  500 , dogs(val) :  500
      ',', 'dogs(val) : ', len(os.listdir(val_dogs_dir)))

# 제너레이터 준비(증강 / 스케일링)
train_datagen = ImageDataGenerator(
    rescale = 1 / 255,
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)

val_datagen = ImageDataGenerator(rescale = 1 / 255)   # val 데이터는 스케일링만 하는 것이 바람직

train_data = train_datagen.flow_from_directory(         # 폴더에 대해 자동 라벨링됨
    train_dir,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',       # categoicl
    shuffle = True,
)

val_data = val_datagen.flow_from_directory(         # 검증의 경우는 굳이 섞기 안함
    validation_dir,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',       # categoicl
    shuffle = False,
)

print('class(label) index : ', train_data.class_indices)    # {'cats': 0, 'dogs': 1}
print('val class(label) index : ', val_data.class_indices)    # {'cats': 0, 'dogs': 1}



imgs, labels = next(train_data)
n_show = min(12, imgs.shape[0])
cols = 6
rows = int(np.ceil(n_show / cols))
idx_to_name = {v:k for k,v in train_data.class_indices.items()}
print(idx_to_name)  # {0: 'cats', 1: 'dogs'} ->

# 시각화
plt.figure(figsize = (10, 2 * rows))
for i in range(n_show):
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(imgs[i])
    ax.set_title(f'{idx_to_name[int(labels[i])]}')
    ax.axis('off')

plt.suptitle('sample train image', fontsize = 14)
plt.tight_layout()
plt.show()

# model
model = Sequential([
    Input((IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(16, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(256, activation = 'relu'),
    Dropout(0.3),
    Dense(1, activation = 'sigmoid'),
])

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

os.makedirs('chkpoints', exist_ok = True)
ckpt = ModelCheckpoint(
    filepath = 'chkpoints/catdog.keras',
    monitor = 'val_accuracy',
    mode = 'max',
    save_best_only = True,
    verbose = 2
)
es = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True)

history = model.fit(train_data, epochs = EPOCHS,
                    validation_data = val_data, callbacks = [ckpt, es], verbose = 2)
val_loss, val_acc = model.evaluate(val_data, verbose = 0)

print(f'acc : {val_acc:.4f}, loss : {val_loss:.4f}')

# 평가 시각화
plt.figure(figsize = (12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'train_acc')
plt.plot(history.history['val_accuracy'], label = 'val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (12,4))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()

# 검증 배치 예측
preview_gen = ImageDataGenerator(rescale = 1./255)
preview_flow = preview_gen.flow_from_directory(
    validation_dir,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = 24,
    class_mode = 'binary',      # categorical
    shuffle = True,
    seed = 123
)

# 예측용 개/고양이 6장 읽기
n_each = 6      # 고양이 / 개 각각 n개 모일 때까지 여러 배치 이어받기
cats_imgs, dogs_imgs = [], []

while len(cats_imgs) < n_each or len(dogs_imgs) < n_each:
    imgs, labels = next(preview_flow)   # 섞인 배치
    for im, lb in zip(imgs, labels.ravel()):    # im - image, lb - label
        if lb == 0 and len(cats_imgs) < n_each: # label 이 1이면 강아지, 0이면 고양이
            cats_imgs.append(im)
        elif lb == 1 and len(cats_imgs) < n_each:
            dogs_imgs.append(im)
        if len(cats_imgs) >= n_each and len(dogs_imgs) >= n_each:
            break

# 예측
cats_probs = model.predict(np.array(cats_imgs), verbose = 0).ravel()
dogs_probs = model.predict(np.array(dogs_imgs), verbose = 0).ravel()

print(cats_probs)
print(dogs_probs)

# 예측 결과 시각화
rows, cols = 2, n_each
plt.figure(figsize = (3 * cols, 5))
for i in range(n_each):
    # cats row
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(cats_imgs[i]);
    ax.axis('off')
    p = cats_probs[i]
    ax.set_title(f"True : cats|Pred : {'dogs' if p >= 0.5 else 'cats'}(p_dog={p:.2f})", fontsize = 9)

    # dogs row
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(dogs_imgs[i]);
    ax.axis('off')
    p = dogs_probs[i]
    ax.set_title(f"True : dogs|Pred : {'dogs' if p >= 0.5 else 'cats'}(p_dog={p:.2f})", fontsize = 9)

plt.suptitle('validation preview', fontsize = 12)
plt.tight_layout()
plt.show()

# 새 이미지를 분류 예측
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = 'chkpoints/catdogmodel.keras'
IMG_HEIGHT, IMG_WEIGHT = 150, 150

THRESH = 0.5

idx_to_name = {0:'cats', 1:'dogs'}

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# 전처리
def preprocess_img(img_path):
    # 단일 이미지 경로를 받아 (1, 150, 150, 3) 텐서로 변환해줘야만 한다
    img = tf.keras.utils.load_img(img_path, target_size = (IMG_HEIGHT, IMG_WEIGHT))
    arr = tf.keras.utils.img_to_array(img)      # H, W, C   float32
    arr = arr / 255.0   # scaling 하는거야 지금
    arr = np.expand_dims(arr, axis = 0)     # 차원 추가 -> (1, Height, Width, Channel)
    return arr

def predict_one(img_path, show = True):
    # 이미지 하나를 분류예측하고 출력 후 반환
    x = preprocess_img(img_path)    # 전처리

    prob_dog = float(model.predict(x, verbose = 0)[0][0])   # sigmoid 출력이라 dog 확률을 뽑는건가봐
    pred_idx = int(prob_dog >= THRESH)
    pred_name = idx_to_name[pred_idx]

    prob_cat = 1 - prob_dog     # cat 확률

    # 단일 이미지 시각화
    if show:
        img_disp = tf.keras.utils.load_img(img_path, target_size = (IMG_HEIGHT, IMG_WEIGHT))
        plt.figure(figsize = (4, 4))
        plt.imshow(img_disp)
        plt.axis('off')
        plt.title(f'pred : {pred_name} | p(cat) = {prob_cat:.2f}, p(dog) = {prob_dog:.2f}')
        plt.show()

    return {'path' : img_path, 'pred' : pred_name, 'p_cat' : prob_cat, 'p_dog' : prob_dog}

result = predict_one('myimg.jpg', show = True)
print('결과는 ', result)