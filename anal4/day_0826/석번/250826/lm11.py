'''
## 선형회귀 평가 지표 관련
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # 학습용(train)과 테스트용(test)으로 나누는 함수

# 공부 시간에 따른 시험 점수 : 표본 수 16
df = pd.DataFrame({'studytime':[3,4,5,8,10,5,8,6,3,6,10,9,7,9,0,2],
                   'score':[76,74,74,89,66,75,84,82,73,81,89,88,83,70,40,69]}) # 표본 16개
# print(df.head(3))

# dataset을 train / test data로 분리한다. - 절대 sort하면 안된다.(왜곡된 자료로 분리하면 안된다. 시계열 데이터는 논외)
train, test = train_test_split(df, test_size=0.2, random_state=1) 
# train 0.8, test 0.2으로 매번 랜덤하게 분할된다.(비복원)
# random_state : random.seed(1)과 같은 역할이다.
# print(len(train), len(test)) # 12 4

'''
## sklearn 입출력 차원 규칙
- 입력, 독립변수 x(feature) : 2차원 배열이어야 한다.
    - 변환 방법 : reshape(-1, 1) ...
    - 모양 : (n_samples, n_features)

- 출력, 종속변수 y(target) : 1차원 벡터가 기본이다.
    - 모양 : (n_samples,)
'''
x_train = train[['studytime']] # 독립변수는 2차원
y_train = train['score'] # 종속변수는 1차원

# print(x_train)
# print(y_train)

x_test = test[['studytime']]
y_test = test['score']

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape,)
# (12, 1) (4, 1) (12,) (4,)

model = LinearRegression()
model.fit(x_train, y_train) # 모델 학습은 train data를 사용한다.
y_pred = model.predict(x_test) # 모델 평가용 예측은 test data를 사용한다.

print(f'예측값 : {np.round(y_pred, 0)}')
print(f'실제값 : {y_test.values}')
# 예측값 : [81. 84. 76. 73.]
# 실제값 : [89 70 82 74]


# 모델 성능 평가 - 결정계수 r2_score, MSE가 일반적이다.
# - 결정계수 수식으로 직접 작성 후 api 메소드와 비교
# 1. 잔차 residual ; 평균값 필요
y_mean = np.mean(y_test) # y의 평균

# (분자) 오차 제곱합 : y실제값 - y예측값 의 제곱 합
bunja = np.sum(np.square(y_test - y_pred))

# (분모) 편차 제곱합 : y관측값 - y평균값 의 제곱 합
bunmo = np.sum(np.square(y_test - y_mean))

# 결정계수 R2
r2 = 1 - bunja/bunmo # 1 - SSE/SST
print(f'계산에 의한 결정계수 : {r2}') # -0.21500261349464722

# 2. 모듈
from sklearn.metrics import r2_score
print(f'api 제공 메소드 결정계수 : {r2_score(y_test, y_pred)}') # -0.36929415023803447

# R2값은 분산을 기반으로 측정하는 도구인데 중심극한정리에 의해 표본 데이터가 많아지면 그 수치도 증가한다.
# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

def linearFunc(data, test_size,k):
    print(f'------------함수 실행{k+1}번째--------------')
    train, test = train_test_split(df, train_size=test_size, shuffle=True, random_state=2)
    x_train = train[['studytime']]
    y_train = train['score']
    x_test = test[['studytime']]
    y_test = test['score']

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # R2 계산
    print(f'R2 값 : {r2_score(y_test, y_pred):.2f}')
    print(f'test data 비율 : 전체 데이터 수의 {test_size * 100}%')
    print(f'데이터 수 : {len(x_train)}개')

    # 시각화
    sns.scatterplot(x=df['studytime'], y=df['score'], color='green')
    sns.scatterplot(x=x_test['studytime'], y=y_test, color='red')
    sns.lineplot(x=x_test['studytime'], y=y_pred, color='blue')   
    plt.show()
    plt.close()

test_sizes = [0.1,0.2,0.3,0.4,0.5] # test 자료 수를 10~50%로 늘려가며 R2 값 구하기
for k,i in enumerate(test_sizes):
    linearFunc(df, i,k)