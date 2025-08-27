## LinearRegression으로 선형회귀 모델 작성 - mtcars

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
import statsmodels.api
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
# print(mtcars.head(3))
# print(mtcars.corr(method='pearson'))

# 데이터 추출
x = mtcars[['hp']].values # 마력 2차원 배열로 추출
y = mtcars['mpg'].values
# print(x[:3])
# print(y[:3])

# LinearRegression
lmodel = LinearRegression().fit(x,y)
print(f'slope : {lmodel.coef_[0]:.4f}') # 기울기 : -0.0682
print(f'intercept : {lmodel.intercept_:.4f}') # 절편 : 30.0989

# 시각화
'''
plt.scatter(x,y)
plt.plot(x, lmodel.coef_ * x + lmodel.intercept_, c='r') # 추세선
plt.show()
plt.close()
'''

# 예측
pred = lmodel.predict(x)
print(f'예측값 : {np.round(pred[:5], 1)}')
print(f'실제값 : {y[:5]}')

# 모델 성능 평가
print(f'r2_score(결정계수): {r2_score(y, pred):.4f}') # 0.6024 => 60.24% 설명해주고 있다.
print(f'MSE(평균제곱오차): {mean_squared_error(y, pred):.4f}') # 13.9898

# 새로운 마력 수에 대한 연비는?
new_hp = [[123]]
new_pred = lmodel.predict(new_hp)
print('%.4f 마력인 경우 연비는 약 %.4f 입니다'%(new_hp[0][0], new_pred[0]))