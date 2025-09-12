'''
문제2)
https://github.com/pykwon/python/tree/master/data
자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟`수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.
새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력하시오
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv',parse_dates=['datetime'])
# print(data.head(3))
data.info()
print(data.isna().sum())

data['year'] = data['datetime'].dt.year # 연월일시분초 칼럼 생성
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['second'] = data['datetime'].dt.second

print(data.head(3))

data.drop(['casual','registered', 'minute','second'], axis=1, inplace=True)

# 상관관계 확인
corr = data.corr(numeric_only=True).round(2)
print(corr)
plt.figure(figsize=(16,9))
sns.heatmap(corr, annot=True, cmap='Blues')
plt.tight_layout()
plt.show()
plt.close()

sns.pairplot(data[['count', 'temp','atemp','humidity','hour']], diag_kind='kde')
plt.show()

# feature 선정 : 'temp','atemp','humidity','hour'
features = data[['temp','atemp','humidity','hour']]
label = data[['count']]

# train / test split
x_train, x_test, y_train, y_test = train_test_split(features, label, train_size=0.3, random_state=23)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 표준화
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.fit_transform(x_test)
print(x_train_sc[:2], x_test_sc[:2])

# 모델
def build_model():
    model = Sequential([
        Input(shape=x_train_sc.shape[1:],),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear'),
    ])
    opti = optimizers.Adam(learning_rate=0.001)
    model.compile(opti, loss='mse', metrics=['mse'])

    return model

model = build_model()
print(model.summary())

# 조기 종료
early_stop = EarlyStopping(monitor='val_loss', baseline=0.03, patience=5)

# 학습
history = model.fit(x_train_sc, y_train, batch_size=32, epochs=5000, validation_split=0.2, callbacks=[early_stop])

# 평가 evaluate
loss, mse = model.evaluate(x_test_sc, y_test)
print('test dataset으로 평가 loss : {:5.3f}'.format(loss))
print('test dataset으로 평가 mse : {:5.3f}'.format(mse))

print('결정계수 : ', r2_score(y_test, model.predict(x_test_sc)))

# 시각화
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.figure(figsize=(10,6))
plt.plot(hist['epoch'], hist['mse'], label='train err')
plt.plot(hist['epoch'], hist['val_mse'], label='validation err')
plt.xlabel('epochs')
plt.ylabel('mse [count]')
plt.show()
plt.close()


# 새로운 값으로 예측
temp_input = float(input('temp input :'))
atemp_input = float(input('atemp input : '))
humidity_input = int(input('humidity input :'))
hour_input = int(input('hour input : '))

new_data = pd.DataFrame({'temp' : [temp_input], 'atemp':[atemp_input], 'humidity':[humidity_input], 'hour':[hour_input]})

new_data_sc = scaler.fit_transform(new_data)
new_pred = model.predict(new_data_sc).ravel()
print('예측 결과 : ', new_pred)