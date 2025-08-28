## 보스턴 집값 데이터를 이용해 단순, 다향회귀 모델 작성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data', 
                 header=None, sep=r'\s+') # sep 정규식 s(공백) +(여러개)
# 열 이름
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'] # LSTAT, MEDV 쓸 예정

# 잘 가져왔는지 체크
# print(df.head(2))

# 상관관계 확인 => -0.7376 음의 상관관계를 가졌다
# print(np.corrcoef(df['LSTAT'], df['MEDV'])[0][1])

# 데이터 추출
x = df[['LSTAT']].values # 하위계층 비율
y = df['MEDV'].values # 주택가격 중앙값

# 선형모델
model = LinearRegression()

# 비선형모델
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

# 데이터 준비
x_quad = quad.fit_transform(x) # 열 2개
x_cubic = cubic.fit_transform(x) # 열 3개

# 단순회귀
model.fit(x,y)
## 시각화 준비
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)
print(y_lin_fit.shape)
# - r2
model_r2 = r2_score(y, model.predict(x))
print(f'model_r2 : {model_r2:.4f}') # 0.5441

## 시각화
'''
plt.scatter(x,y, label='학습데이터', color='lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(1차원), $R^2=%.2f$'%(model_r2), color='blue', lw=3)
plt.xlabel('하위계층 비율')
plt.ylabel('주택가격')
plt.legend()
plt.show()
'''

# 다항(2차)회귀
model.fit(x_quad, y)
## 시각화 준비
y_quad_fit = model.predict(quad.fit_transform(x_fit))
q_r2 = r2_score(y, model.predict(x_quad))
print(f'q_r2 : {q_r2:.4f}') # 0.6407


# 다항(3차)회귀
model.fit(x_cubic, y)
## 시각화 준비
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
c_r2 = r2_score(y, model.predict(x_cubic))
print(f'c_r2 : {c_r2:.4f}') # 0.6578

## 시각화
plt.scatter(x,y, label='학습데이터', color='lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(1차원), $R^2=%.2f$'%(model_r2), color='blue', lw=1)
plt.plot(x_fit, y_quad_fit, linestyle='-', label='quad fit(2차원), $R^2=%.2f$'%(q_r2), color='red', lw=2)
plt.plot(x_fit, y_cubic_fit, linestyle=':', label='cubic fit(3차원), $R^2=%.2f$'%(c_r2), color='black', lw=3)
plt.xlabel('하위계층 비율')
plt.ylabel('주택가격')
plt.legend()
plt.show()

# 다항회귀 오버피팅 주의.

print(x_fit.shape)
print(y_lin_fit.shape)