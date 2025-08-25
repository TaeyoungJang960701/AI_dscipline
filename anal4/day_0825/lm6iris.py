# 단순선형회귀 : ols 사용
# 상관관계가 선형회귀모델에 미치는 영향에 대해

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head(4))
print('-' * 100)
print(iris.iloc[:, 0:4].corr())
print('-' * 100)

# 연습 1 : 상관관계가 약한(-0.117570) 두 변수를(sepal_width, sepal_length) 사용
result1 = smf.ols(formula = 'sepal_length ~ sepal_width', data = iris).fit()
print('검정 결과 1 :', result1.summary())
print('결정 계수 1 : ', result1.rsquared)        # 0.0138 이건 설명력이 너무낮아
print('pvalue : ', result1.pvalues[1])          # 0.15189826071144785 이건 0.05보다 크다
# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color = 'r')
# plt.show()      # 이건 너무 흩어져잇고 설명력도 낮아서 쓸 수가 없는 데이터야

# 연습 2 : 상관관계가 강한(0.871754) 두 변수(petal_length, sepal_length) 사용
result2 = smf.ols(formula = 'sepal_length ~ petal_length', data = iris).fit()
print('검정 결과 2 :', result2.summary())
print('결정 계수 2 : ', result2.rsquared)        # 0.6690276860464137
print('pvalue : ', result2.pvalues.iloc[1])
# plt.scatter(iris.petal_length, iris.sepal_length)
# plt.plot(iris.petal_length, result2.predict(), color = 'r')
# plt.show()

# 일부의 실제값과 예측값 비교
print('실제값 : ', iris.sepal_length[:5].values)
print('예측값 : ', result2.predict()[:5])
print('-' * 100)


new_data = pd.DataFrame({'petal_length' : [1.1,0.5,5.0]})
y_pred = result2.predict(new_data)
print('예측결과(sepal_length) : \n', y_pred)

print('-- 다중 선형회귀 : 독립변수 복수 --')
# result3 = smf.ols(formula = 'sepal_length ~ petal_length + petal_width + sepal_width', 
#                   data = iris).fit()
# 리절트3 에서는 sepal_length가 종속변수고 독립변수는 petal_length, petal_width, sepal_width
column_select = '+'.join(iris.columns.difference(['sepal_length','species']))
result3 = smf.ols(formula = 'sepal_length ~ ' + column_select, data = iris).fit()
print(result3.summary())
# -- 다중 선형회귀 : 독립변수 복수 --
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:           sepal_length   R-squared:                       0.859
# Model:                            OLS   Adj. R-squared:                  0.856    설명수준 85.6% 
# Method:                 Least Squares   F-statistic:                     295.5    아주 좋은 데이터야
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           8.59e-62
# Time:                        11:47:40   Log-Likelihood:                -37.321
# No. Observations:                 150   AIC:                             82.64
# Df Residuals:                     146   BIC:                             94.69
# Df Model:                           3
# Covariance Type:            nonrobust
# ================================================================================
#                    coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------
# Intercept        1.8560      0.251      7.401      0.000       1.360       2.352
# petal_length     0.7091      0.057     12.502      0.000       0.597       0.821
# petal_width     -0.5565      0.128     -4.363      0.000      -0.809      -0.304
# sepal_width      0.6508      0.067      9.765      0.000       0.519       0.783  
# ==============================================================================
# Omnibus:                        0.345   Durbin-Watson:                   2.060
# Prob(Omnibus):                  0.842   Jarque-Bera (JB):                0.504
# Skew:                           0.007   Prob(JB):                        0.777
# Kurtosis:                       2.716   Cond. No.                         54.7
# ==============================================================================