'''
## 비선형회귀분석
- 선형관계분석의 경우 모델에 다항식 또는 교호작용이 있는 경우에는 해석이 덜 직관적이다.
- 결과의 신뢰성이 떨어진다.
- 선형가정이 어긋날 때(정규성 위배) 대처하는 방법으로 다항식 항을 추가한 다항회귀 모델을 작성할 수 있다.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
'''
plt.scatter(x,y)
plt.show() # 모양이 선형으로 선을 긋기는 애매하다.
print(np.corrcoef(x,y)[0][1]) # 0.4807
'''

# 선형회귀 모델 작성
from sklearn.linear_model import LinearRegression
x = x[:,np.newaxis] # 2차원으로 만들어주기 위해 차원 확대(reshape 방법도 있음)
# print(x)
model1 = LinearRegression().fit(x,y)
y_pred = model1.predict(x)
print(f'예측값1 : {y_pred}')
print(f'결정계수1 : {r2_score(y,y_pred)}') # 0.23113

# 시각화
'''
plt.scatter(x,y)
plt.plot(x,y_pred, color='red')
plt.show()
'''

# 다항회귀 모델 작성 - 추세선의 유연성을 위해 열 추가
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False) # degree=열수
x2 = poly.fit_transform(x) # 특징 행렬을 만들기 <= 다향을 만들기 위함
# print(x2) # 5행 3열로 만들었다.

model2 = LinearRegression().fit(x2,y) # 특징 행렬로 학습
y_pred2 = model2.predict(x2)
print(f'예측값2 : {y_pred2}') 
# [4.14285714 1.62857143 1.25714286 3.02857143 6.94285714] vs [4,2,1,3,7]
print(f'결정계수2 : {r2_score(y,y_pred2)}') # 0.9892
# - 오버피팅에 유의해야 한다.

# 시각화
plt.scatter(x,y)
plt.plot(x,y_pred2, color='blue')
plt.show() # degree를 늘리면 plot 더 붙는다. 오버피팅 유의.