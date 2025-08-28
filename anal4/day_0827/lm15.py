import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([257, 270, 294, 320,342,368,396,446,480,580])[:,np.newaxis]  # 1차원이야
print(x.shape)
y = np.array([236,234,253,298,314,342,360,368,390,388])
# plt.scatter(x,y)
# plt.show()

# 일반회귀모델과 다항회귀모델 작성 후 비교
lr = LinearRegression()
pr = LinearRegression()

polyf = PolynomialFeatures(degree = 2)
x_quad = polyf.fit_transform(x)

# 일반 회귀모델
lr.fit(x,y)
pr.fit(x_quad,y)   # 다항 회귀모델 학습 추가
x_fit = np.arange(250,600,10)[:,np.newaxis]

# 예측값 계산
y_lin_fit = lr.predict(x_fit)  # 선형은 예측 구간에 대해 예측해야 함
y_quad_fit = pr.predict(polyf.transform(x_fit))  # 다항 예측은 변환 후 예측

plt.scatter(x,y, label = 'training point')
plt.plot(x_fit, y_lin_fit, label = 'linear fit', linestyle = '--', c = 'red')
plt.plot(x_fit, y_quad_fit, label = 'quadratic fit', linestyle = '-.', color = 'blue')
# plt.legend()
# plt.show()

# 성능 비교 점수
print('MSE : 선형:%.3f, 다항:%.3f'%(
      mean_squared_error(y, lr.predict(x)),\
      mean_squared_error(y, pr.predict(x_quad)) ))

print('설명력 : 선형:%.3f, 다항:%.3f'%(
      r2_score(y, lr.predict(x)), \
      r2_score(y, pr.predict(x_quad)) ))

# 보스턴 집값으로 실습
# 