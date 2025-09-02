# [Randomforest 문제1] 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
# dataset은 winequality-red.csv 
# https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv


# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

df1 = pd.read_csv('winequality-red.csv')
print(df1.head())
x = df1.drop('quality', axis = 1)
y = df1['quality']

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42).fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(f'와인 등급 메기기 모델 정확도 : {round(accuracy_score(y_test,y_pred), 2)}')
print('예측된 와인 등급 : ', y_pred[:10])
print('실제 와인 등급 : ', y_test.values[:10])

# [Randomforest 문제2]
# 중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료다.
# 종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오. 

# 예제 파일 : https://github.com/pykwon  ==>  patient.csv
# <변수설명>
#   STA : 환자 생사 여부 (0:생존, 1:사망)
#   AGE : 나이
#   SEX : 성별
#   RACE : 인종
#   SER : 중환자 치료실에서 받은 치료
#   CAN : 암 존재 여부
#   INF : 중환자 치료실에서의 감염 여부
#   CPR : 중환자 치료실 도착 전 CPR여부
#   HRA : 중환자 치료실에서의 심박수

from sklearn import preprocessing

df2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/patient.csv')
print(df2.head())
print(df2['RACE'].unique())
df2.drop(columns = ['ID'], inplace = True)
print(df2.head())

x = df2.drop(columns = ['STA'])
y = df2['STA']
# print(df2.isnull().sum())

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier().fit(x_train, y_train)  # 문제랑 정답지 같이 줘서 학습해
pred = model.predict(x_test)
print('실제값 : ', y_test)
print('예측값 : ', pred)
print('정확도 : ', accuracy_score(y_test, pred))

