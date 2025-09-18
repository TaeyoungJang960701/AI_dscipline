# MNIST dataset으로 CNN 모델 작성
# Sequential Api, Functional Api

# i) Sequential Api
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 구조 변경(차원)
print(x_train.shape)        # (60000, 28, 28)
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_train.shape)        # (60000, 28, 28, 1)
# print(x_train[0])

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (28, 28, 1)),  # shape = (사진크기, 채널)
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    tf.keras.layers.Dropout(rate = 0.2),

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    # stride 이거 (1,1)일때는 그냥 생략해도 된대
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),   # 간혹 가다가 이 줄 안넣은 사람들 모델도 보인대

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),

    tf.keras.layers.Flatten(),   # Fully Connected Layer -> 1차원으로 축소
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dropout(rate= 0.3),

    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dropout(rate= 0.3),

    tf.keras.layers.Dense(units = 10, activation = 'softmax')
    ])
# 이 모델에서 가장 중요한거 - CNN에서 가장 중요한건 MaxPool과 Flatten, 그리고 Dense 이 세 클래스

print(model.summary())

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

es = tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs = 100, batch_size = 128, validation_split = 0.1,
                    callbacks = [es], verbose = 2
                    )

# 모델 평가
train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0)
print(f'train_loss : {train_loss:.4f} train_accuracy : {train_acc:.4f}')
print(f'test_loss : {test_loss:.4f} test_accuracy : {test_acc:.4f}')

# 모델 저장
save_path = '/content/drive/MyDrive/mysou/mnist_cnn.keras'
model.save(save_path)

# 모델 읽기

loaded_model = tf.keras.models.load_model(save_path)
loss2, acc2 = loaded_model.evaluate(x_test, y_test, verbose = 0)
print(f'loss2 : {loss2:.4f} accuracy : {acc2:.4f}')

# 기존 자료 1개로 예측
idx = 0     # 해보고싶은거 아무거나

x_one = x_test[idx:idx + 1]     # (1, 28, 28, 1)
y_true = int(y_test[idx])

probs = loaded_model.predict(x_one, verbose = 0)[0]     # (10.)
y_pred = int(np.argmax(probs))
print(f'실제값 : {y_true}, 예측값 : {y_pred}, 예측 확률값 : {np.round(probs, 3)}')

# 시각화 : 학습 곡선 (정확도 / 손실)
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
plt.close()

# 단일 이미지 + 예측 확률 막대 시각화
classes = [str(i) for i in range(10)]
print(classes)

plt.figure(figsize = (10,4))

plt.subplot(1,2,1)

plt.imshow(x_one[0].squeeze(), cmap = 'gray')

plt.axis('off')
plt.title(f'True : {y_true} | Pred : {y_pred}')

plt.subplot(1,2,2)
plt.bar(classes, probs)
plt.title('Prediction Probabilities')
plt.xlabel('class')
plt.ylabel('probability')
plt.ylim(0,1.0)
for i, v in enumerate(probs):
    plt.text(i, v + 0.2, f'{v:.2f}', ha = 'center', fontsize = 9)

plt.tight_layout()
plt.show()
plt.close()

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_all = np.argmax(loaded_model.predict(x_test, verbose = 0), axis = 1)
print(y_pred_all)
cm = confusion_matrix(y_test, y_pred_all, labels = list(range(10)))
disp = ConfusionMatrixDisplay(cm, display_labels = classes)
fig, ax = plt.subplots(figsize = (6, 6))
disp.plot(ax = ax, cmap = 'Blues', values_format = 'd', colorbar = False)

plt.title('Confusion matrix')
plt.tight_layout()
plt.show()
plt.close()

# --- preprocess + predict for CNN(MNIST) ---
from PIL import Image, ImageOps
# ImageOps : ImageOps는 Pillow(PIL) 의 유틸 모듈. 이미지에 자주 쓰는 후처리/기하 변환/톤 조정 기능들을 한 곳에 모아둔 것
import numpy as np
import tensorflow as tf

def preprocess_mnist(path, invert="auto"):
    im = Image.open(path).convert("L")

    try:
        im = ImageOps.pad(im, (28, 28), method=Image.Resampling.LANCZOS, color=255, centering=(0.5, 0.5))
    except AttributeError:
        im = ImageOps.pad(im, (28, 28), color=255, centering=(0.5, 0.5))

    img = np.asarray(im).astype("float32")  # (28, 28), 0..255

    # MNIST는 "검은 배경(0), 흰 글씨(255)". 배경이 밝으면 자동 반전
    if invert == "auto":
        if img.mean() > 127:
            img = 255.0 - img
    elif invert is True:
        img = 255.0 - img

    # 정규화 + 채널 차원 추가 → (1, 28, 28, 1)
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

save_path = "mnist_cnn.keras"
mymodel = tf.keras.models.load_model(save_path)

# 내 이미지 전처리 & 예측
data = preprocess_mnist('su.png')             # shape (1, 28, 28, 1)
new_pred = mymodel.predict(data, verbose=0)   # shape (1, 10)

pred_class = int(np.argmax(new_pred, axis=1)[0])
pred_conf  = float(new_pred[0, pred_class])
print('probs  :', np.round(new_pred, 4))
print('class  :', pred_class)
print('conf   :', pred_conf)

