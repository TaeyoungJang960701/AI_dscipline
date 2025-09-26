# 백본(backbone) 모델 : MobileNetV2
# 희귀한 소량의 이미지 데이터는 cifar10 데이터로 대신함
# MobileNetV2 모델 그대로 학습시켜
# 내 이미지 데이터를 잘 분류하는 모델 생성

# Transfer Learning, Fine-tunning

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape)    # (50000, 32, 32, 3)
num_classes = 10

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Removed the redundant one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Transfer Learning(전이학습) : 기존 모델의 가중치는 모두 동결(Freeze)
# ->새로 추가한 분류층만 학습
base_model = keras.applications.MobileNetV2(
    input_shape = (96,96,3),
    include_top = False,    # 기본 분류기
    # 분류기 부분을 빼고 컨벌루션(Conv2D같은거)만 해당, 분류기(Dense) 부분만 학습에 참여한대
    weights = 'imagenet'
)

base_model.trainable = False        # 이게 동결시키는거야

# 함수형 api로 모델 생성
inputs = keras.Input(shape = (32,32,3))
x = layers.Resizing(96,96)(inputs)
x = base_model(x, training = False)
x = layers.GlobalAveragePooling2D()(x)      # MaxPooling보다 더 급격하게 feature의 크기를 줄인다
outputs = layers.Dense(units = num_classes, activation = 'softmax')(x)
model_tl = keras.Model(inputs, outputs)

model_tl.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Use the corrected y_train and y_test
model_tl.fit(x_train, y_train, epochs = 5, validation_split = 0.1, batch_size = 64, verbose = 1)

print('모델 test 평가 결과 : ', model_tl.evaluate(x_test,y_test))

# Fine-Tunning (미세조정)
# 베이스 모델(백본) 일부 층만 열기(예 : 마지막 30개)

base_model.trainable = True        # 이게 동결시키는거야

for layer in base_model.layers[:-30]:
    layer.trainable = False

# 이제 낮은 학습률(learning rate)로 재 컴파일
model_tl = keras.Model(inputs, outputs)

model_tl.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-5),
                 loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_tl.fit(x_train, y_train, epochs = 5, validation_split = 0.1, batch_size = 64, verbose = 1)

print('모델 test 평가 결과 : ', model_tl.evaluate(x_test,y_test))

