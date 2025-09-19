from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv', header=None, names=column_names)
print(data.head())
print(data.info())
x_data = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y_data = data[['Outcome']]

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.3, random_state=12, shuffle=True, stratify=y_data)
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape) # (537, 8) (231, 8) (537, 1) (231, 1)
print(x_train[:5])
print(y_train[:5])

# 모델 생성 Sequential API 방식
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2)) # 과적합 방지용
model.add(BatchNormalization()) # 배치 정규화, 역전파시 기울기 소실 또는 폭주 방지, CNN에서 효과적임
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(rate=0.1)) # 과적합 방지용
model.add(BatchNormalization())
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam' ,loss='binary_crossentropy', metrics=['accuracy'])
# fit() 전에 model score 확인
loss, acc = model.evaluate(x_train, y_train , verbose=0)
print('훈련 전 모델 정확도{:5.2f}%'.format(100 * acc)) # 훈련 전 모델 정확도 34.82%

early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train, y_train, validation_split= 0.2, epochs=1000, batch_size=64,callbacks=[early_stop],verbose=2)

loss, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('훈련 후 모델 정확도{:5.2f}%'.format(100 * acc)) # 훈련 후 모델 정확도 69.70%

# 시각화
epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len, history.history['val_loss'], label = 'val_loss')
plt.plot(epoch_len, history.history['loss'], label = 'loss', c='red')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc= 'best')
plt.show()

plt.plot(epoch_len, history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(epoch_len, history.history['accuracy'], label = 'accuracy', c='red')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc= 'best')

# 모델 정의 방법 2: Functional api
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

inputs = Input(shape=(8,))
x = Dense(units=32, activation='relu')(inputs)
x = Dropout(rate=0.2)(x)
x = BatchNormalization()(x)
x = Dense(units=16, activation='relu')(x)
x = Dropout(rate=0.1)(x)
x = BatchNormalization()(x)
x = Dense(units=8, activation='relu')(x)
outputs = Dense(units=1, activation='sigmoid')(x)
model2 = Model(inputs=inputs, outputs=outputs)
model2.summary()
model2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01),metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10)
loss, acc = model2.evaluate(x_train, y_train , verbose=0)
print('훈련 전 모델 정확도{:5.2f}%'.format(100 * acc)) # 훈련 전 모델 정확도33.71%
history2 = model2.fit(x_train, y_train, validation_split= 0.2, epochs=1000, batch_size=64,callbacks=[early_stop],verbose=2)

loss, acc = model2.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('훈련 후 모델 정확도{:5.2f}%'.format(100 * acc)) # 훈련 후 모델 정확도68.40%

# 시각화
epoch_len = np.arange(len(history2.epoch))

plt.plot(epoch_len, history2.history['val_loss'], label = 'val_loss')
plt.plot(epoch_len, history2.history['loss'], label = 'loss', c='red')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc= 'best')
plt.show()

plt.plot(epoch_len, history2.history['val_accuracy'], label = 'val_accuracy')
plt.plot(epoch_len, history2.history['accuracy'], label = 'accuracy', c='red')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc = 'best')