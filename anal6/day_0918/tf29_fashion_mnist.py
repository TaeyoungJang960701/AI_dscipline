# Fashion MNIST dataset으로 이미지 분류 모델
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top','Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
print(train_images.shape)   # (60000, 28, 28)
print(test_images.shape)    # (10000, 28, 28)
print(set(train_labels))

plt.imshow(train_images[0], cmap = 'Greys')
plt.show()

# 25개의 이미지 확인
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
    plt.imshow(train_images[i])
plt.show()

# 데이터값 조정
train_images = train_images / 255.0
test_images = test_images / 255.0
# print(train_images[0])

# model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),    # 완전(밀집) 연결 층
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(units = 10, activation = 'softmax'),
     ])
print(model.summary())

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_images, train_labels, batch_size = 64, epochs = 5, verbose = 1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy : ', test_acc)
print('Test loss : ', test_loss)

# 예측
pred = model.predict(test_images)
print('예측값 : \n', np.argmax(pred[0]))
print('실제값 : \n', test_labels[0])

# 각 이미지 출력용 함수 (예측 이미지와 실제 레이블을 비교 판별)
def plot_image(i, pred_arr, true_label_arr, img_arr):
    pred_arr = pred_arr[i]
    true_label = true_label_arr[i]
    img = img_arr[i]

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap = 'Greys')

    pred_label = np.argmax(pred_arr)
    # 예측값과 실제값이 같으면 blue, 다르면 red로 표시
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[pred_label],
                                         100 * np.max(pred_arr), class_names[true_label]), color = color)

i = 40

plt.figure(figsize = (8,6))
plt.subplot(1,2,1)
plot_image(i, pred, test_labels, test_images)

def plot_value_arr(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i]
    thisplot = plt.bar(range(10), pred_arr)
    plt.ylim([0,1])
    pred_label = np.argmax(pred_arr)
    thisplot[pred_label].set_color('red')   # 예측값
    thisplot[true_label].set_color('blue')  # 실제값

plt.subplot(1,2,2)
plot_value_arr(i, pred, test_labels)
plt.show()

