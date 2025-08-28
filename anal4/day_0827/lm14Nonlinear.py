# 비선형회귀분석

# 선형관계분석의 경우 모델에 다항식 또는 교호작용이 있는 경우에는
# 해석이 덜 직관적이다.  ====> 결과에 신뢰성이 떨어진다는 얘기

# 선형 가정이 어긋날 때(정규성 위배) 대처하는 방법으로 다항식 항을 추가한 다항회귀 모델을 작성할 수 있다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

print(np.corrcoef(x,y))

# 선형회귀 모델 작성
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]    # 차원을 확대하는 과정
# print(x)
model1 = LinearRegression().fit(x,y)

ypred = model1.predict(x)
print('예측값 : ', ypred)
print('실제값 : ', y)
print('결정계수1 : ', r2_score(y,ypred))

# plt.scatter(x,y)
# plt.plot(x,ypred,color = 'red')
# plt.show()

# 다항회귀 모델 작성 - 추세선의 유연성을 위해 열을 추가한다
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 5, include_bias = False)     
# 디그리 = 열(column)의 수,     bias => 상수항을 포함할지 안할지. False니까 안한대
x2 = poly.fit_transform(x)      # 특징 행렬을 만듦
print(x2)
# [[ 1.  1.]
#  [ 2.  4.]
#  [ 3.  9.]
#  [ 4. 16.]
#  [ 5. 25.]]
# 12345의 x 벡터가 2차원의 행렬이 되었다.

model2 = LinearRegression().fit(x2,y)
ypred2 = model2.predict(x2)
print('예측값 : ', ypred2)
print('결정계수 : ', r2_score(y,ypred2))    
# 0.9892183288409704 이건 데이터 풀이 아주 작아서 정말 높게 나왓어 근데 전혀 좋은게 아냐
# 오버피팅되었거든
# degree를 2에서 올리니까 점점 더 결정계수가 올라가는데 이건 마냥 좋은게 아니야

plt.scatter(x,y)
plt.plot(x,ypred2,color = 'blue')
plt.show()

# linear하게 각 점끼리 잇기는 해주는데 따르는 차수의 그래프가 올라감에 따라 결정계수가 올라가는걸 보여줫어