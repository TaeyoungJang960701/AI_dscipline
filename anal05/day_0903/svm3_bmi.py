# BMI 식을 이용해 데이터 만들기
# 비만도 계산은 몸무게를 키의 제곱으로 나눈 것
# 예) 키 : 175, 몸무게 : 68     ===> 68 / (170 / 100)^2

# 아래는 csv 파일을 그냥 만들기 위한 코드엿어

"""
import random
random.seed(12)

def calc_bmi(h,w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return 'thin'
    elif bmi <25 : return 'normal'
    return 'fat'

print(calc_bmi(170,68))     # normal
print(calc_bmi(170,98))     # fat

fp = open('bmi.csv', mode = 'w')
fp.write('height,weight,label\n')

cnt = {'thin' : 0, 'normal' : 0, 'fat' : 0}

for i in range(50000):
    h = random.randint(150,200)
    w = random.randint(35, 100)
    label = calc_bmi(h,w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h,w,label))

fp.close()
"""

# SVM으로 분류 모델
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tbl = pd.read_csv('bmi.csv')
print(tbl.head())
print(tbl.info())

label = tbl['label']
print(label.head())

# 정규화 한거야 normalize 한거지 그니까 1 이하의 수로 데이터를 통일한거야 
# 원래는 최대값으로 나눠야 정확한 정규화지만 연습이니까 넘어가자

w = tbl['weight'] / 100
print(w[:5].values)
h = tbl['height'] / 200
print(h[:5].values)
print('-' * 100)

wh = pd.concat([w,h], axis = 1)
print(wh.head(),wh.shape)
print('-' * 100)

# label를 정수화
label = label.map({'thin' : 0, 'normal' : 1, 'fat' : 2})
print(label[:5])
print('-' * 100)

x_train, x_test, y_train, y_test = \
    train_test_split(wh,label,test_size=0.3, random_state=1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = svm.SVC(C=0.01, kernel = 'rbf', random_state=1).fit(x_train, y_train)     
# C값이 커지면 margin의 폭이 좁아지면서 과적합이 될 가능성이 있대
# C값은 규제 강도거든

pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)

ac_score = metrics.accuracy_score(y_test, pred)
print('분류 정확도 : ', ac_score)

tbl2 = pd.read_csv('bmi.csv', index_col = 2)
print(tbl2)

def scatterFunc(lbl, color):
    b = tbl2.loc[lbl]
    # plt.scatter(b['weight'], b['height'], c = color, label = lbl)

scatterFunc('fat', 'red')
scatterFunc('normal', 'yellow')
scatterFunc('thin', 'blue')
# plt.legend()
# plt.show()

# 새로운 값으로 예측
newData = pd.DataFrame({'weight' : [19, 89], 'height' : [170,170]})
newData['weight'] = newData['weight'] / 100
newData['height'] = newData['height'] / 200
# 위 세줄은 정규화한거엿어

new_pred = model.predict(newData)
print('새로운 데이터에 대한 bmi는 ', new_pred)