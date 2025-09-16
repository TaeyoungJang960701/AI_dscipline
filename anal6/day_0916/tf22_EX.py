# 문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
# * 변수 종류 *
# satisfaction_level : 직무 만족도
# last_evaluation : 마지막 평가점수
# number_project : 진행 프로젝트 수
# average_monthly_hours : 월평균 근무시간
# time_spend_company : 근속년수
# work_accident : 사건사고 여부(0: 없음, 1: 있음)
# left : 이직 여부(0: 잔류, 1: 이직)
# promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
# sales : 부서
# salary : 임금 수준 (low, medium, high)

# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.


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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/HR_comma_sep.csv')

# 방법 i) 원 핫 인코딩으로 샐러리의 스트링 밸류들을 벡터(매트릭스)화
le = LabelEncoder()
y = le.fit_transform(df['salary'])
y = to_categorical(y)

# 방법 ii) 그냥 맵핑으로 샐러리를 단순 범주화
df['salary'] = df['salary'].map({'low' : 0, 'medium' : 1, 'high' : 2})
x = df.drop('salary', axis=1)
y = df['salary']

print(x.head())
print(y.head())

x = pd.get_dummies(x, columns=['sales'], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

feature_importances = pd.Series(model.feature_importances_, index=x_train.columns)

sorted_feature_importances = feature_importances.sort_values(ascending=False)

print("Important features:")
print(sorted_feature_importances)

plt.figure(figsize=(10, 6))
sorted_feature_importances.plot(kind='bar')
plt.title("Feature Importances (RandomForest)")
plt.ylabel("Importance")
plt.show()

x = df[['average_montly_hours', 'last_evaluation', 'satisfaction_level', 'time_spend_company', 'number_project']]
# 월평균 근무시간(연속형), 마지막 평가점수(연속형), 직무 만족도(연속형), 근속년수(연속형), 진행 프로젝트수(연속형)
y = df[['salary']]

print(x.head(),x.shape)     # (14999, 5)
print(y.head(),y.shape)     # (14999, 1)

nb_classes1 = y.shape[1]
nb_classes2 = len(set(y))

# train / test

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2, random_state = 42, stratify = y)



# model
# 방법 i와 방법 ii에 따라 model1, model2 로 나눠 씀
model1 = Sequential([
    Input(shape = (x.shape[1],)),
    Dense(64, activation = 'relu'),
    Dropout(0.3),
    Dense(32, activation = 'relu'),
    Dense(nb_classes1, activation = 'softmax')
])

model2 = Sequential([
    Input(shape = (x.shape[1],)),
    Dense(64,activation = 'relu'),
    Dropout(0.3),
    Dense(units = 32, activation = 'relu'),
    Dense(len(set(y_train.values.ravel())), activation = 'softmax'),
])

model1.summary()
model2.summary()

model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model2.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# callback
early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)

checkpoint = ModelCheckpoint('best_zoom_model.keras', monitor = 'val_loss', save_best_only = True)

history1 = model1.fit(x_train, y_train, epochs = 1000, validation_split = 0.2,
                    callbacks = [early_stop, checkpoint], verbose = 1)

history2 = model2.fit(x_train, y_train, epochs = 1000, validation_split = 0.2,
                    callbacks = [early_stop, checkpoint], verbose = 1)

loss1, acc1 = model1.evaluate(x_test, y_test, verbose = 0)
print(f'방법 i) 최종 평가 Loss : {loss1:.2f}, Accuracy : {acc1:.2f}')

loss2, acc2 = model2.evaluate(x_test, y_test, verbose = 0)
print(f'방법 ii) 최종 평가 Loss : {loss2:.2f}, Accuracy : {acc2:.2f}')


# 학습 곡선 시각화
# loss 곡선
plt.plot(history1.history['loss'], label = 'train_loss')
plt.plot(history1.history['val_loss'], '--', label = 'val loss')
plt.title('method i : one-hot encoding')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

# accuracy 곡선
plt.plot(history1.history['accuracy'], label = 'train_accuracy')
plt.plot(history1.history['val_accuracy'], '--', label = 'val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()

# loss 곡선
plt.plot(history2.history['loss'], label = 'train_loss')
plt.plot(history2.history['val_loss'], '--', label = 'val loss')
plt.title('method ii : simply mapped')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

# accuracy 곡선
plt.plot(history2.history['accuracy'], label = 'train_accuracy')
plt.plot(history2.history['val_accuracy'], '--', label = 'val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.close()

