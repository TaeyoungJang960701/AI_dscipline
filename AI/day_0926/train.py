import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)

DATA_DIR_TRAIN = os.path.join(BASE_DIR,'train')
# print(DATA_DIR_TRAIN)
DATA_DIR_VAL = os.path.join(BASE_DIR,'validation')
# print(DATA_DIR_TRAIN)

IMG_SIZE = (224,224)
BATCH = 32
EPOCHS = 30
LR = 1e-3

# train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    image_size = IMG_SIZE,
    batch_size = BATCH,
    shuffle = True,
    seed = SEED     # 트레인 데이터를 섞어주는것도 과적합 방지의 한 방법이래
)

# validation dataset        검증 데이터는 순서를 유지해준다
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_VAL,
    image_size = IMG_SIZE,
    batch_size = BATCH,
    shuffle = False,
    seed = SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
# print('Classes : ', class_names, ' ', num_classes)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)    # train데이터만 섞고
val_ds = val_ds.cache().prefetch(AUTOTUNE)                      # validation 데이터는 섞지 않아

# 데이터 변형 또는 증강
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),    # 이미지 좌우반전이 일어남
    layers.RandomRotation(0.05),        # 소량 회전
    layers.RandomZoom(0.1),             # 소량을 줌 땡기기
])

preprocess = mobilenet_v2.preprocess_input      # [-1, 1] 범위로 스케일링

# 백본(backbone model) 구성
base = mobilenet_v2.MobileNetV2(
    include_top = False,
    weights = 'imagenet',
    input_shape = IMG_SIZE + (3,)
)
base.trainable = False

# 나의 모델 생성
inputs = keras.Input(shape = IMG_SIZE + (3,))
x = data_augmentation(inputs)       # 입력에 증강 적용
x = layers.Lambda(preprocess)(x)
x = base(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation = 'softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer = keras.optimizers.Adam(learning_rate = LR),
              loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor = 'val_accuracy', mode = 'max',
        save_best_only = True, verbose =2 
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy', mode = 'max',
        patience = 3, restore_best_weights = True
    ),
]

history = model.fit(
    train_ds, validation_data = val_ds, epochs = EPOCHS, callbacks = callbacks,

    )

val_loss , val_acc = model.evaluate(val_ds, verbose = 0)
print(f'acc : {val_acc:.4f}, loss = {val_loss:.4f}')

# 미세 조정(Fine-tunning)
unfreeze_from = 100

for layer in base.layers[unfreeze_from:]:
    layer.trainable = True

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
              loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
              )

print(model.summary())

fine_history = model.fit(
    train_ds, validation_data = val_ds, epochs = EPOCHS, callbacks = callbacks,
)

val_loss , val_acc = model.evaluate(val_ds, verbose = 0)
print(f'final acc : {val_acc:.4f}, final loss = {val_loss:.4f}')

with open('class_name.txt', mode = 'w', encoding = 'utf-8') as f:
    for name in class_names:
        f.write(f'{name}\n')