# zoo dataset으로 다항 분류 모델 작성
# animal_name: Unique for each instance
# hair Boolean
# feathers Boolean
# eggs Boolean
# milk Boolean
# airborne Boolean
# aquatic Boolean
# predator Boolean
# toothed Boolean
# backbone Boolean
# breathes Boolean
# venomous Boolean
# fins Boolean
# legs Numeric (set of values: {0,2,4,5,6,8})
# tail Boolean
# domestic Boolean
# catsize Boolean
# class_type Numeric (integer values in range [1,7])

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
# from tensorflow.keras.utils import to_categorical   # one-hot encoding 지원
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/zoo.csv'
datas = pd.read_csv(url)
print(datas.head())
x_data = datas.iloc[:, :-1].astype('float32').values
y_data = datas.iloc[:, -1].astype('int32').values
print(x_data[:2], x_data.shape)     # (101, 16)
print(y_data[:2], y_data.shape)     # (101,)
nb_classes = len(set(y_data))
print('classes 범주 : ', nb_classes)

# train / test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 42, stratify = y_data)

# model
model = Sequential([
    Input(shape = (x_data.shape[1],)),
    Dense(64,activation = 'relu'),
    Dropout(0.3),
    Dense(32, activation = 'relu'),
    Dense(nb_classes, activation = 'softmax'),
])
model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# callback
early_stop =EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)
# restore_best_weights = True   =>  학습 종료 후 가장 좋은 val_loss를 기록한 epoch의 가중치

checkpoint = ModelCheckpoint('best_zoom_model.keras', monitor = 'val_loss', save_best_only=True)
history = model.fit(x_train, y_train, epochs = 1000, validation_split = 0.2,
                    callbacks = [early_stop, checkpoint], verbose = 1)
loss, acc = model.evaluate(x_test, y_test, verbose = 0)
print(f'최종 평가 : Loss -> {loss:.4f}, Accuracy : {acc:.4f}')

# 학습 곡선 시각화
plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], '--', label = 'val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], '--', label = 'val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()


# confusion matrix & Report
# 먼저 Report
y_pred = np.argmax(model.predict(x_test), axis = 1)
print(y_pred)
print('Report : ', classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Acual')
plt.show()

# best 모델 읽기
from tensorflow.keras.models import load_model
best_model = load_model('best_zoom_model.keras')

loss, acc = best_model.evaluate(x_test, y_test, verbose = 0)
print(f'최종평가 : Loss -> {loss:.4f}, Accuracy : {acc:.4f}')

# 새로운 데이터 분류
new_data = np.array([[1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 52., 0., 0., 1.]])
probs = best_model.predict(new_data)
print(probs)
pred_class = np.argmax(probs)
print('예측 결과 : ', pred_class)