# 백본(backbone) 모델 : MobileNetV2
# 희귀한 소량의 이미지 데이터는 cifar10 데이터로 대신함
# MobileNetV2 모델 그대로 학습시켜 내 이미지 데이터를 잘 분류하는 모델 생성import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape)    # (50000, 32, 32, 3)
num_classes = 10

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('train data shape : ', x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 10)

# MobileNetV2 모델 호출
mobilenet_model = keras.applications.MobileNetV2(
    # MobileNetV2 : 입력 최소크기 : 32, 권장크기 : 96, 128, 160, 192, 디폴트는 224
    input_shape = (32,32,3),    # MobileNetV2
    include_top = True,         # 기본 분류기 포함
    weights = None,             # 랜덤 초기화 (imagenet X)
    classes = num_classes       # 내 이미지 데이터 클래스를 적용 (10개)
)

# print(mobilenet_model.summary())
mobilenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = mobilenet_model.fit(x_train, y_train,
              epochs = 2, batch_size = 64, validation_split = 0.25, verbose = 2)
print('test 평가 결과 : ', mobilenet_model.evaluate(x_test,y_test))

