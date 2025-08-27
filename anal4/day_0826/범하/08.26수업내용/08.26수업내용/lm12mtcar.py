# LinearRegression으로  선형회귀 모델 작성 - mtcar

import statsmodels.api
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

mtcars=statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars.corr(method='pearson'))
print()
x=mtcars[['hp']].values
print(x[:3])
y=mtcars['mpg'].values
print(y[:3])

lmodel=LinearRegression().fit(x,y)
print('slope : ',lmodel.coef_)              # slope :  -0.06822828
print('intercept : ',lmodel.intercept_)     # intercept :  30.098860539622496
plt.scatter(x,y)
plt.plot(x,lmodel.coef_*x+lmodel.intercept_,c='r')
# plt.show()

pred=lmodel.predict(x)
print('예측값 : ',np.round(pred[:5],1))
print('예측값 : ',y[:5]) 
print()

# 모델성능평가
print('MSE : ', mean_squared_error(y,pred)) # MSE :  13.989822298268805
print('r2_score : ',r2_score(y,pred))       # r2_score :  0.602437341423934

# 새로운 마력수에 대한 연비는?
new_hp=[[123]]
new_pred=lmodel.predict(new_hp)
print('%s 마력인 경우 연비는 약 %s입니다'%(new_hp[0][0],new_pred[0]))
# 123 마력인 경우 연비는 약 21.70678233682017입니다
