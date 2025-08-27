'''
## sklearn 모듈의 LinearRegression 클래스 사용
- summary 없다. 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
from sklearn.linear_model import LinearRegression # 안쓰고 ols 쓰면 된다. 
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error # 성능 확인 (결정계수...)
from sklearn.preprocessing import StandardScaler, MinMaxScaler # 표준화 / 정규화 ; RobustScaler은 강경한 방법이다.

# 표본 데이터
# 1. 편차가 없는 데이터 생성
sample_size = 100
np.random.seed(1)

x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
# print(x[:5]) # [ 16.24345364  -6.11756414  -5.28171752 -10.72968622   8.65407629]
# print(y[:5])

# 상관계수 확인
# print(np.corrcoef(x,y)[0,1]) # 0.9998478075508624

# x 정규화
# - 정규화하지 않고 학습시키면 모델이 좋지 않게 나올 수 있다. 그래서 데이터 평탄화 작업해준다.
# - scaling은 모델 기능을 상향 시키는 방법 중 하나이다.
# - scaling은 독립변수만 한다.
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1,1))
# print(x_scaled)

# 시각화
'''
plt.scatter(x_scaled, y)
plt.show()
plt.close()
'''

# 모델
model = LinearRegression().fit(x_scaled,y)
# print(f'계수(slope) : {model.coef_}') # 회귀계수 (각 독립변수가 종속변수에 미치는 영향)
# 1350.4161554

# print(f'절편(intercept) : {model.intercept_}')
# -691.1877661754081

# print(f'결정계수(r2) : {model.score(x_scaled,y)}') # 설명력 : 훈련 데이터 기준
# 0.9987875127274646
# y = wx + b <= y = 1350.4161554 * x + -691.1877661754081
# 식 대신 predict 쓸 수 있다

y_pred = model.predict(x_scaled)
# - summary() 함수 지원하지 않는다.

# print(f'예측값(y^): {y_pred[:5]}')
# print(f'실제값(y): {y[:5]}')
# 예측값(y^): [ 490.32381062 -182.64057041 -157.48540955 -321.44435455  261.91825779]
# 실제값(y): [ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]

'''
## 모델의 성능 평가 지표
- MAE(Mean Absolute of Errors) 평균절대오차
- MSE(Mean Square of Errors) 평균제곱오차
- RMSE(Root Mean Square of Errors) 평균제곱오차제곱근
- R2(R Squared Score) 결정계수
'''
# 모델 성능 파악용 함수 작성
def RegScoreFunc(y_true, y_pred): # r2_score, explained_variance_score, mean_squared_error 써보자
    print(f'R2_score(결정계수): {r2_score(y_true, y_pred)}')
    print(f'설명분산점수: {explained_variance_score(y_true, y_pred)}') # 실제값, 예측값 순
    print(f'MSE(평균제곱오차): {mean_squared_error(y_true, y_pred)}') # 숫자가 작을수록 좋다. 하지만 기준이 없다.(유동적)

# RegScoreFunc(y,y_pred)
# R2_score(결정계수): 0.9987875127274646 가독성이 좋아서 더 많이 사용한다.
# 설명분산점수: 0.9987875127274646
# MSE(평균제곱오차): 86.14795101998747

# 2. 편차가 있는 데이터 생성
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print(x[:5]) # [-0.40087819  0.82400562 -0.56230543  1.95487808 -1.33195167]
print(y[:5])

# 상관계수 확인
# print(f'상관계수 : {np.corrcoef(x,y)[0,1]}') # 0.004011673780558856

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1,1))
# print(x_scaled)

# 시각화
plt.scatter(x_scaled, y)
plt.show()
plt.close()
'''
'''

# 모델
mode2 = LinearRegression().fit(x_scaled,y)

y_pred = mode2.predict(x_scaled)
print(f'예측값(y^): {y_pred[:5]}')
print(f'실제값(y): {y[:5]}')
# 예측값(y^): [-10.75792685  -8.15919008 -11.10041394  -5.7599096  -12.73331002]
# 실제값(y): [1020.86531436 -710.85829436 -431.95511059 -381.64245767 -179.50741077]

# 모델 성능 평가
RegScoreFunc(y,y_pred)
# R2_score(결정계수): 1.6093526521765433e-05
# 설명분산점수: 1.6093526521765433e-05
# MSE(평균제곱오차): 282457.9703485092
# 편차가 크고 MSE가 굉장히 크며, 결정계수가 매우 작아 좋은 모델이라고 볼 수 없다.