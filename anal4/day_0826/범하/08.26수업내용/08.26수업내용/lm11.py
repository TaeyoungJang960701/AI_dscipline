# 선형회귀 평가 지표 관련

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 공부 시간에 따른 시험성적 데이터 생성 : 표본 수 16
df=pd.DataFrame({'studytime':[3,4,5,8,10,5,8,6,3,6,10,9,7,0,1,2],'score':[76,74,74,89,92,75,84,82,73,81,89,88,83,40,70,69]})
print(df.head(3))

# dataset분리 : train/test data - sort 절대 X(왜곡된 자료로 분리 하면 X)
train,test=train_test_split(df,test_size=0.4,random_state=1)
print(len(train),len(test)) # 8 8
x_train=train[['studytime']]
y_train=train['score']
x_test=test[['studytime']]
y_test=test['score']
print(x_train)
# 14          1
# 0           3
# 15          2
# 9           6
# 8           3
# 12          7
# 11          9
# 5           5
print(y_train)
# 14    70
# 0     76
# 15    69
# 9     81
# 8     73
# 12    83
# 11    88
# 5     75
# Linear Regression은 1차원만 사용가능하기에 위와 같은 형태로 변환함
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# (9, 1) (7, 1) (9,) (7,)
print()
model=LinearRegression()
model.fit(x_train,y_train)  # 모델학슨은 train data를 사용
y_pred=model.predict(x_test)    # 모델 평가용 예측은 test data를 사용
print('예측값 : ',np.round(y_pred,0))   # 예측값 :  [85. 66. 80. 78. 85. 90. 90.]
print('실제값 : ',y_test.values)        # 실제값 :  [89 40 82 74 84 89 92]

print()
print('모델의 성능은? - r2_score,MSE가 일반적이다')
# 결정계수 수식으로 직접 작성 후 api메소드와 비교
# 잔차 구하기
y_mean=np.mean(y_test)  # y의 평균
# 오차 제곱합 : sum(y실제값 - y예측값)^2
bunja=np.sum(np.square(y_test-y_pred))
# 편차 제곱합 : sum(y실제값 - y평균값)^2
bunmo=np.sum(np.square(y_test-y_mean))
r2=1-bunja/bunmo    # 1-(오차제곱합/편차제곱합)
print('계산에 결정계수 : ',r2)  # 계산에 결정계수 :  0.628675204713166

from sklearn.metrics import r2_score
print('api 제공 메서드 결정계수 : ',r2_score(y_test,y_pred))    # api 제공 메서드 결정계수 :  0.628675204713166

# R^2같은 분산을 기반으로 측정하는 도구인데 중심극한정리에 의해 표본데이터가 많아지면 그 수치도 증가한다
import seaborn as sns
import matplotlib.pyplot as plt

def linearFunc(df,test_size):
    train,test=train_test_split(df,test_size=test_size,shuffle=True,random_state=2)
    x_train=train[['studytime']]
    y_train=train['score']
    x_test=test[['studytime']]
    y_test=test['score']

    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    # R^2계산
    print('R^2값 : ',r2_score(y_test,y_pred))
    print('test data비율 : 전체 데이터의 수의 {0}%'.format(test_size*100))
    print('데이터의 수 : {0}개'.format(x_train))
    # 시각화 
    sns.scatterplot(x=df['studytime'],y=df['score'],color='green')
    sns.scatterplot(x=x_test['studytime'],y=y_test,color='red')
    sns.lineplot(x=x_test['studytime'],y=y_pred,color='blue')
    plt.show()

# test자료수를  10%에서 50%로 늘려가며 R^2값 구하기
test_size=[0.1,0.2,0.3,0.4,0.5]  
for i in test_size:
    linearFunc(df,i)










