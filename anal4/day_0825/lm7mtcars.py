# 선형회귀분석 - ols() 사용
# mtcars 데이터셋을 사용 - 독립변수가 종속변수(mpg, 연비)에 영향을 미치는가?

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
# print(mtcars)
# print(mtcars.shape)

print(mtcars.columns)   # ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']
print(mtcars.info())
# print(mtcars.corr())

print(np.corrcoef(mtcars.hp, mtcars.mpg)[0,1])      # -0.7761683718265864
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0,1])      # -0.8676593765172281
# 음의 상관관계가 아주 뚜렷해 절대값이 1에 아주 근사하다

# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('hp')
# plt.ylabel('mpg')
# plt.show()

print('\n단순 선형회귀')
result = smf.ols(formula = 'mpg ~ hp', data = mtcars).fit()
print(result.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                    mpg   R-squared:                       0.602    설명력도 양호해
# Model:                            OLS   Adj. R-squared:                  0.589
# Method:                 Least Squares   F-statistic:                     45.46
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.79e-07
# Time:                        12:14:15   Log-Likelihood:                -87.619
# No. Observations:                  32   AIC:                             179.2
# Df Residuals:                      30   BIC:                             182.2
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     30.0989      1.634     18.421      0.000      26.762      33.436
# hp            -0.0682      0.010     -6.742      0.000      -0.089      -0.048
# ==============================================================================
# Omnibus:                        3.692   Durbin-Watson:                   1.134
# Prob(Omnibus):                  0.158   Jarque-Bera (JB):                2.984
# Skew:                           0.747   Prob(JB):                        0.225
# Kurtosis:                       2.935   Cond. No.                         386.
# ==============================================================================

print('마력수 110에 대한 연비 예측 : ', -0.0682 * 110 + 30.0989)    # 22.5969
print('마력수 50에 대한 연비 예측 : ', -0.0682 * 50 + 30.0989)      # 26.6889
print('마력수 110에 대한 연비 예측 : ',result.predict(pd.DataFrame({'hp' : [110]})))    # 22.59375
print('마력수 50에 대한 연비 예측 : ',result.predict(pd.DataFrame({'hp' : [50]})))      # 26.687447

print('\n다중선형회귀')
result2 = smf.ols(formula = 'mpg ~ hp + wt', data = mtcars).fit()
print(result2.summary())



