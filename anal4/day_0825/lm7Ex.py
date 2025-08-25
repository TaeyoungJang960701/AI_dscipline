import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns

# 회귀분석 문제 2) 
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측

student = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
print(student)

x = student.국어
y = student.수학

model1 = smf.ols(formula = '수학 ~ 국어', data = student).fit()
print(model1.summary())

intercept = model1.params[0]
slope = model1.params[1] 
pred_math = slope * x + intercept

plt.scatter(x,y)
plt.plot(x,pred_math,'r')
# plt.show()

# 국어점수로 수학점수 예측해보기
# korean = int(input('국어 점수를 입력하세요\n:'))
# math = slope * korean + intercept
# print(f'예상 수학 점수는 {math} 점입니다')

y = student.수학
x1 = student.국어
x2 = student.영어
# ols 다변수로 쓰면 될듯

print(np.corrcoef(x1,y)[0,1])   # 0.7662626557853176
print(np.corrcoef(x2,y)[0,1])   # 0.8096677295287106 
# 두 데이터 모두 종속변수에 주는 영향이 의미가 있어보임

result = smf.ols(formula = '수학 ~ 국어 + 영어', data = student).fit()
print(result.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     수학   R-squared:                       0.659
# Model:                            OLS   Adj. R-squared:                  0.619
# Method:                 Least Squares   F-statistic:                     16.46
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           0.000105
# Time:                        14:35:49   Log-Likelihood:                -74.617
# No. Observations:                  20   AIC:                             155.2
# Df Residuals:                      17   BIC:                             158.2
# Df Model:                           2
# Covariance Type:            nonrobust <<<<<<-------------- 이거는 데이터가 fit 하지 못하단 뜻인가봐
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     22.6238      9.482      2.386      0.029       2.618      42.629
# 국어             0.1158      0.261      0.443      0.663      -0.436       0.667
# 영어             0.5942      0.313      1.900      0.074      -0.066       1.254  t값은 0.05를 넘어 어떡하지 이건
# ==============================================================================
# Omnibus:                        6.313   Durbin-Watson:                   2.163
# Prob(Omnibus):                  0.043   Jarque-Bera (JB):                3.824
# Skew:                          -0.927   Prob(JB):                        0.148
# Kurtosis:                       4.073   Cond. No.                         412.
# ==============================================================================

x1 = int(input('국어 점수를 입력하세요 \n:'))
x2 = int(input('영어 점수를 입력하세요 \n:'))
pred_math = 22.6238 + x1 * 0.1158 + x2 * 0.5942

print(f'예상 수학 점수는 {pred_math} 점입니다.')
# 정확도는 그렇게 좋지 않아 데이터값들이 좀 튀어서 정확하지 못하게 나왓나봐