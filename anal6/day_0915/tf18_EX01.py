# 문제1) https://www.kaggle.com/jyotikumarrout/graduation 의 binary.csv 데이터를 이용하여
# 미국 대학원 입학여부를 분류하는 모델을 작성하시오.
# loss, accuracy에 대한 시각화도 실시한다.
# input 함수를 사용해 새로운 gre, gpa, rank 값을 받아  admit을 판정하시오.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("graduation_data/binary.csv")
print(df.head())

x = df.drop('admit', axis=1).values
y = df['admit'].values

print(x[:5])
print(y[:5])

np.set_printoptions(suppress = True)
print('-' * 100)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle = True, random_state=12, stratify = y)

print("\nx_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model = Sequential()

model.add(Input(shape = (x_train.shape[1],)))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(BatchNormalization())
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(BatchNormalization())
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
print(model.summary())

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
loss, acc = model.evaluate(x_train, y_train, verbose = 0)

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = 'model/{epoch:02d}_val_loss_{val_loss:.4f}.keras'

chkpoint = ModelCheckpoint(
    filepath = modelpath,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True
)

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5
)

history = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = 1000,
    callbacks = [early_stop, chkpoint],
    verbose = 2
)

loss, acc = model.evaluate(x_test, y_test, batch_size = 64, verbose = 0)
print('훈련 후 모델 정확도 : {:5.2f}%'.format(100*acc))

epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len,history.history['val_loss'],label='val_loss')
plt.plot(epoch_len,history.history['loss'],label='loss',c='red')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

plt.plot(epoch_len,history.history['val_acc'],label='val_acc')
plt.plot(epoch_len,history.history['acc'],label='acc',c='red')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(loc='best')
plt.show()

from types import new_class
# best모델로 예측
from tensorflow.keras.models import load_model
model=load_model(MODEL_DIR+'43_0.0667.keras')

new_data=x_test[:5,:]
print(new_data)
pred=model.predict(new_data)
print('예측결과 : ',np.where(pred>=0.5,1,0).ravel())

# Separate features (X) and target (y)
# Exclude the 'admit' column for features
x = df.drop('admit', axis=1).values
# Use the 'admit' column as the target
y = df['admit'].values

print("First 5 rows of features (x):")
print(x[:5])
print("\nFirst 5 rows of target (y):")
print(y[:5])

np.set_printoptions(suppress = True)
print('-' * 100)

# Split data into training and testing sets
# Use stratify=y to maintain the same proportion of admit/not admit in both sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle = True, random_state=12, stratify = y)

print("\nx_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model = Sequential()
# Correct the input shape to match the number of features in x_train (which is 3)
model.add(Input(shape = (x_train.shape[1],))) # Use x_train.shape[1] to dynamically get the number of features
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(BatchNormalization())
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(BatchNormalization())
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
print(model.summary())

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
loss, acc = model.evaluate(x_train, y_train, verbose = 0)

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# Correct the modelpath string format
modelpath = 'model/{epoch:02d}_val_loss_{val_loss:.4f}.keras'

chkpoint = ModelCheckpoint(
    filepath = modelpath,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True
)

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5
)

history = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = 1000,
    callbacks = [early_stop, chkpoint],
    verbose = 2
)

loss, acc = model.evaluate(x_test, y_test, batch_size = 64, verbose = 0)
print('훈련 후 모델 정확도 : {:5.2f}%'.format(100*acc))

epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len,history.history['val_loss'],label='val_loss')
plt.plot(epoch_len,history.history['loss'],label='loss',c='red')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

plt.plot(epoch_len,history.history['val_acc'],label='val_acc')
plt.plot(epoch_len,history.history['acc'],label='acc',c='red')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(loc='best')
plt.show()

from types import new_class
# best모델로 예측
from tensorflow.keras.models import load_model

# Find the best model file based on validation loss
best_model_file = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')])[-1]
best_model_path = os.path.join(MODEL_DIR, best_model_file)

model=load_model(best_model_path)

new_data=x_test[:5,:]
print(new_data)
pred=model.predict(new_data)
print('예측결과 : ',np.where(pred>=0.5,1,0).ravel())

def predict_admission(model):
    gre = float(input("Enter GRE score: "))
    gpa = float(input("Enter GPA: "))
    rank = int(input("Enter rank (1, 2, 3, or 4): "))

    new_data = np.array([[gre, gpa, rank]])

    prediction = model.predict(new_data)

    if prediction[0][0] >= 0.5:
        print(f"\nPrediction: Admit (Probability: {prediction[0][0]:.4f})")
    else:
        print(f"\nPrediction: Not Admit (Probability: {prediction[0][0]:.4f})")

predict_admission(model)

