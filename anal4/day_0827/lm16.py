# 보스턴 집값 데이터를 이용, 다항 회귀 모델 작성

# 독립변수 : 모집단의 하위계층의 비율
# 종속변수 : 본인 소유의 주택가격

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rc('font', family = 'Malgun Gothic')

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data', header = None, sep = r'\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS',
              'NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT',
              'MEDV']
print(df.head())
print(df.corr())            # MEDV와 LSTAT 사이 상관계수는 -0.737663 아주 큰 상관관계가 잇어보임
x = df[['LSTAT']].values    # 하위계층 비율
y = df[['MEDV']].values     # 주택 가격 중앙값

model = LinearRegression()

quad = PolynomialFeatures(degree = 2, include_bias=False)
cubic = PolynomialFeatures(degree = 3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)

# 단순회귀
model.fit(x,y)
x_fit = np.arange(x.min(), x.max()+1, 1)[:, np.newaxis] 
y_lin_fit = model.predict(x_fit)
# print(y_lin_fit)

model_r2 = r2_score(y,model.predict(x))
print('model_r2 : ', model_r2)

# 다항(2차)
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.transform(x_fit))  # x_fit 기준 예측
q_r2 = r2_score(y, model.predict(x_quad))          # 학습 데이터 기준 R²
print('q_r2 : ', q_r2)          # q_r2 :  0.6407168971636612

# 다항(3차)
model.fit(x_cubic,y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
c_r2 = r2_score(y, model.predict(x_cubic))
print('c_r2 : ', c_r2)          # c_r2 :  0.6578476405895719

# 시각화
plt.scatter(x,y, label = '학습데이터', c = 'lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit (d=1), R²=%.2f' % model_r2, c='b', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle='-.', label='quad fit (d=2), R²=%.2f' % q_r2, c='r', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='cubic fit (d=3), R²=%.2f' % c_r2, c='k', lw=3)
plt.xlabel('하위계층비율')
plt.ylabel('주택가격')
plt.legend()
plt.show()

# 오버피팅을 피하면서, 분석가가 몇 차식을 쓸지를 결정하여 적당한 차수를 선택하여 분석한다.