# Sslearn 모듈의 linearregression 클래스 사용
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,explained_variance_score,mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler   # 표준화(StandardScaler),정규화(minmax_scale)
import matplotlib.pylab as plt  

sample_size=100
np.random.seed(1)   # NumPy의 난수 생성기 시드(seed)를 1로 고정 // 시드(seed)가 1, 난수 생성기를 “초기 상태 #1”로 설정

# 1) 편차가 없는 데이터 생성
x=np.random.normal(0,10,sample_size)
y=np.random.normal(0,10,sample_size)+x*30
print(x[:5])
print(y[:5])
print('상관계수 : ',np.corrcoef(x,y))   # 상관계수 : 0.99984781

scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x.reshape(-1,1))
print('x_scaled : ',x_scaled)
# plt.scatter(x_scaled,y)
# plt.show()

model=LinearRegression().fit(x_scaled,y)
print(model)
print('계수(slope) : ',model.coef_) # 회귀계수(독립변수가 종속변수에 미치는 영향)
print('절편(intercept) : ',model.intercept_)
print('결정계수(R^2) : ',model.score(x_scaled,y))   # 설명력 : 훈련 데이터 기준
# 계수(slope) :  [1350.4161554]
# 절편(intercept) :  -691.1877661754081
# 결정계수(R^2) :  0.9987875127274646
# y=ax+b <=4161554*x-691.1877661754081
y_pred=model.predict(x_scaled)
print('예측값(y^) : ',y_pred[:5]) # 예측값(y^) :  [ 490.32381062 -182.64057041 -157.48540955 -321.44435455  261.91825779]
print('실제값(y^) : ',y[:5])      # 실제값(y^) :  [ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]
# model.summary() 지원 X

print()
# 선형회귀분석의 주요 평가지표로는 R-squared (결정계수), MAE (평균 절대 오차), MSE (평균 제곱 오차), RMSE (평균 제곱근 오차)
# 모델 성능 파악용 함수작성 
def RegScoreFunc(y_true,y_pred):
    print('R^2_score(결정계수):{}'.format(r2_score(y_true,y_pred)))
    print('설명분산점수:{}'.format(explained_variance_score(y_true,y_pred)))
    print('mean_squared_error(평균제곱오차):{}'.format(mean_squared_error(y_true,y_pred)))

RegScoreFunc(y,y_pred)
# R^2_score(결정계수):0.9987875127274646
# 설명분산점수:0.9987875127274646
# mean_squared_error(평균제곱오차):86.14795101998747

print('=='*20)

# 2) 편차가 있는 데이터 생성
x=np.random.normal(0,1,sample_size)
y=np.random.normal(0,500,sample_size)+x*30
print(x[:5])
print(y[:5])
print('상관계수 : ',np.corrcoef(x,y))   # 상관계수 : 0.00401167

scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x.reshape(-1,1))
print('x_scaled : ',x_scaled)
# plt.scatter(x_scaled,y)
# plt.show()

model=LinearRegression().fit(x_scaled,y)
y_pred=model.predict(x_scaled)
print(model)
print('예측값(y^) : ',y_pred[:5])   # 예측값(y^) :   [-10.75792685  -8.15919008 -11.10041394  -5.7599096  -12.73331002]
print('실제값(y^) : ',y[:5])        # 실제값(y^) :  [1020.86531436 -710.85829436 -431.95511059 -381.64245767 -179.50741077]

RegScoreFunc(y,y_pred)
# R^2_score(결정계수):1.6093526521765433e-05
# 설명분산점수:1.6093526521765433e-05
# mean_squared_error(평균제곱오차):282457.9703485092