'''
회귀분석 문제 3)
- kaggle.com에서 carseats.csv 파일 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
- 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
- 회귀분석모형의 적절성을 위한 조건도 체크하시오.
- 완성된 모델로 Sales를 예측.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False # 음수 깨짐 방지
import seaborn as sns
import statsmodels.formula.api as smf

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv'
df = pd.read_csv(url, usecols=[0,1,2,3,4,5,7,8])
# df = df.drop([df.columns[6],df.columns[9],df.columns[10]], axis=1)

# print(df.corr())
#                 Sales  CompPrice    Income  Advertising  Population     Price       Age  Education
# Sales        1.000000   0.064079  0.151951     0.269507    0.050471 -0.444951 -0.231815  -0.051955
# CompPrice    0.064079   1.000000 -0.080653    -0.024199   -0.094707  0.584848 -0.100239   0.025197
# Income v      0.151951  -0.080653  1.000000     0.058995   -0.007877 -0.056698 -0.004670  -0.056855
# Advertising v 0.269507  -0.024199  0.058995     1.000000    0.265652  0.044537 -0.004557  -0.033594
# Population   0.050471  -0.094707 -0.007877     0.265652    1.000000 -0.012144 -0.042663  -0.106378
# Price v      -0.444951   0.584848 -0.056698     0.044537   -0.012144  1.000000 -0.102177   0.011747
# Age v        -0.231815  -0.100239 -0.004670    -0.004557   -0.042663 -0.102177  1.000000   0.006488
# Education   -0.051955   0.025197 -0.056855    -0.033594   -0.106378  0.011747  0.006488   1.000000

lmodel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=df).fit()
# print(lmodel.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  Sales   R-squared:                       0.371
# Model:                            OLS   Adj. R-squared:                  0.364
# Method:                 Least Squares   F-statistic:                     58.21
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.33e-38 < 0.05 유의한 모델
# Time:                        16:49:29   Log-Likelihood:                -889.67
# No. Observations:                 400   AIC:                             1789.
# Df Residuals:                     395   BIC:                             1809.
# Df Model:                           4
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      15.1829      0.777     19.542      0.000      13.656      16.710
# Income          0.0108      0.004      2.664      0.008       0.003       0.019
# Advertising     0.1203      0.017      7.078      0.000       0.087       0.154
# Price          -0.0573      0.005    -11.932      0.000      -0.067      -0.048
# Age            -0.0486      0.007     -6.956      0.000      -0.062      -0.035
# ==============================================================================
# Omnibus:                        3.285   Durbin-Watson:                   1.931
# Prob(Omnibus):                  0.194   Jarque-Bera (JB):                3.336
# Skew:                           0.218   Prob(JB):                        0.189
# Kurtosis:                       2.903   Cond. No.                     1.01e+03
# ==============================================================================
# Income, Advertising, Price, Age 모두 < 0.05 

# 작성된 모델 저장 후 읽어서 사용한다. 모델 개발 후 다른 사람들은 lmodel.predict만 할 줄 알면 된다.
# 저장하는 방법
'''
# 1. pickle 모듈 사용
import pickle

# 저장
with open('mymodel.pickle', mode='wb') as obj:
    pickle.dump(lmodel, obj)

# 읽기
with open('mymodel.pickle', mode='rb') as obj:
    pickle.loads(obj)

mymodel.predict('~~')

# 2. joblib 모듈 사용
import joblib

# 저장
joblib.dump(lmodel, 'python_analysis/analysis04/mymodel.model')

# 읽기
mymodel = joblib.load()
mymodel.predict('~~~')
'''

# 선형회귀분석의 기존 가정 충족 조건
df_lm = df.iloc[:,[0,2,3,5,6]]

# 잔차항 구하기
fitted = lmodel.predict(df_lm)
residual = df_lm.Sales - fitted # 잔차
# print(f'residual : {[residual[:3]]}')
# print(f'잔차 평균 : {[np.mean(residual)]}')

# 1. 선형성 : 잔차가 일정하게 분포되어야 한다.
'''
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'}) # lowess 비모수적 추정
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='gray')
plt.title('선형성')
plt.show()
plt.close() # 잔차가 일정하게 분포하기 때문에 선형성 만족이다.
'''

# 2. 정규성 : 잔차항이 정규 분포를 따라야 한다.
# QQ plot 출력 다른 방법
import scipy.stats as stats
'''
sr = stats.zscore(residual)
[x,y],_ = stats.probplot(sr)
sns.scatterplot(x=x,y=y)
plt.plot([-3,3], [-3,3], '--', color='gray')
plt.title('정규성')
plt.show()
plt.close()
'''
# shapiro test
# print(f'shapiro test : {stats.shapiro(residual).pvalue:.4f}')
# shapiro test : 0.2127 > 0.05 이므로 정규성 만족

# 3. 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
# Durbin-Watson: 1.931
# 2에 근사하고 있다. 자기상관이 없다.

import statsmodels.api as sm
# print(f'Durbin-Watson : {sm.stats.stattools.durbin_watson(residual):.4f}')
# Durbin-Watson : 1.9315

# --------------<25-08-26>---------------
# 4. 등분산성 Homoscedasticity 等分散性
# - 독립변수일 때는 확인할 수 없다. 복수일 때 확인할 수 있다.
# - 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.

# 시각화
# 잔차들의 분산은 일정해야 한다. 눈으로 확인하자.
# 선형성 보듯
'''
sr = stats.zscore(residual)
sns.regplot(x=fitted, y=np.sqrt(abs(sr)), lowess=True, line_kws={'color':'red'}) # lowess 비모수적 추정
plt.show()
plt.close()
'''

# 일반적으로 쓰는 방법
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, lmodel.model.exog) # 잔차와 독립변수 관계 확인. 두개 값을 반환한다.
# print(f'통계량 : {bp_test[0]:.4f}\npvalue : {bp_test[1]:.4f}')
# 통계량 : 1.1277
# pvalue : 0.8899 > 0.05 이므로 등분산성 만족.


# 5. 다중공선성
# - 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

# 분산팽창지수 VIF Variance Inflation Factor
# - 연속형의 경우 10을 넘으면 다중공선성을 의심한다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 사용하기 위해서 dataframe 필요하다.

imsidf = df[['Income', 'Advertising', 'Price', 'Age']]
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(imsidf, i) for i in range(imsidf.shape[1])]
# print(vifdf)
#    vif_value
# 0   5.971040 income
# 1   1.993726 advertising
# 2   9.979281 price, 어떤 요소랑 상관관계가 높은가 확인해야 한다. 찾으면 두 요소에 PCA 진행해서 차원축소한다.
# 3   8.267760 age
# 모든 독립변수가 VIF 값이 10 미만이므로 다중공선성 문제가 발생하지 않았다.

## 저장된 모델 읽어 새로운 데이터에 대한 예측
import joblib

ourmodel = joblib.load('python_analysis/analysis04/mymodel.model')
new_df = pd.DataFrame({'Income':[35,63,25], 'Advertising':[6,3,11], 'Price':[105,88,77], 'Age':[33,55,22]})
new_pred = ourmodel.predict(new_df)
print(f'Sales 예측 결과:\n{new_pred}')
# 0     8.664248
# 1     8.507928
# 2    11.296480