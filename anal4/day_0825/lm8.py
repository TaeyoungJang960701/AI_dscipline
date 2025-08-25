# 선형회귀분석 - ols 사용 
# : 회귀분석 선행 조건 충족 확인하는건가?


# -----선형회귀분석의 가정 충분조건-----
# . 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
# . 등분산성 : 그룹간의 분산이 유사해야 한다. 
#               독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

# advertising.csv 사용 : 각 매체의 광고비에 따른 판매량 관련 자료
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf

plt.rc('font', family = 'Malgun Gothic')


advdf = pd.read_csv(
    'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Advertising.csv',
    usecols=[1,2,3,4]
)
print(advdf.head())
print(advdf.shape)                          # (200, 4)
print(advdf.index, ' ',advdf.columns)       # 대충 다 float 데이터라는 얘기
print(advdf.info())

print(advdf.corr())
print('-' * 100)

# 단순선형회귀모델
# x : tv, y : sales
lmodel1 = smf.ols(formula = 'sales ~ tv', data = advdf).fit()
print(lmodel1.params)
print(lmodel1.pvalues)
print(lmodel1.rsquared)
# print(lmodel1.summary())
# print(lmodel1.summary().tables[2])    이게 생각보다 유용할것같애
print('-' * 100)
print('--- 예측하기 ---')
x_part = pd.DataFrame({'tv' : advdf.tv[:3]})
# print(x_part)
print('실제값 : ', advdf.sales[:3])
print('예측값 : ', lmodel1.predict(x_part).values)

print('새 자료로 예측하기 ---')     # 예측 한번 해보자 아래 100 300 500은 한번 넣어보는 인풋값이야
x_new = pd.DataFrame({'tv' : [100,300,500]})
print('새 자료 예측값 : ', lmodel1.predict(x_new).values)

# plt.scatter(advdf.tv, advdf.sales)
# plt.xlabel('tv')
# plt.ylabel('sales')
# y_pred = lmodel1.predict(advdf.tv)
# plt.plot(advdf.tv, y_pred,'r')
# plt.grid()
# plt.show()

print('*** 선형회귀분석 충분조건 ***')
# 잔차항 구하기 - 이게 (예측값과 실제값의 차이) 이니깐
fitted = lmodel1.predict(advdf)
residual = advdf['sales'] - fitted
print('-' * 100)
print('실제값 : ', advdf['sales'][:5].values)
print('예측값 : ',fitted[:5].values)
print('잔차값 : ', residual[:5].values)
print('잔차의 평균값 : ', np.mean(residual))    # 잔차의 평균값 :  -1.4210854715202005e-15
print('-' * 100)

print('1) 정규성 : 잔차가 정규성을 따르는지 확인(정규분포 형태를 띠는지)')
print('-' * 100)
from scipy.stats import shapiro     # 이건 정규성 확인하기 위한 라이브러리
import seaborn as sns
import statsmodels.api as sm        # Quantile-Quantile plot 지원 (이게 큐큐 플랏)

stat, pv = shapiro(residual)
print(f'shapiro-wilk test => 통계량 : {stat:.4f}, p-value : {pv:.4f}')
print('정규성 만족' if pv > 0.05 else '정규성 위배 가능성이 있음')  
# 유의수준 0.05를 넘어야 두 데이터 간의 관계가 있다 라고 생각하는거야 귀무 대립가설 얘기할때랑 똑같네

# 시각화로 확인 Q-Q plot
# sm.qqplot(residual, line = 's')
# plt.title('잔차 Q-Q plot')
# plt.show()          # 그림만 봣을때는 정규성 만족이나 커브를 그려나가는 부분이 좋지 않다.
print('2) 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해')
print('-' * 100)

from statsmodels.stats.diagnostic import linear_reset   # 모형 적합성 확인
reset_result = linear_reset(lmodel1, power = 2, use_f = True)
print(f'Linear_Reset Test : F = {reset_result.fvalue:.4f}, p = {reset_result.pvalue:.4f}')
print('선형성 만족' if reset_result.pvalue > 0.05 else '선형성 위배 가능성 있음')

# 시각화로 확인
# sns.regplot(x = fitted, y = residual, lowess = True, line_kws = {'color':'red'})
# plt.plot([fitted.min(), fitted.max()], [0,0], '--', color = 'gray')
# plt.show()

print('3) 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.')
# 독립성 가정은 잔차 간에 자기상관이 없어야 한다.
# 자기상관 : 회귀분석 등에서 관측된 값과 추정된 값의 차이인 잔차들이 서로 연관되어 있는 상태
# 더빈-왓슨(Durbin-Watson) 검정으로 확인
print(lmodel1.summary())
# Durbin-Watson : 1.935   => 2에 근사하면 자기상관이 없음

# 참고 : Cook's distance
# 하나의 관측치가 전체 모델에 얼마나 영향을 주는지 수치화한 지표

from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lmodel1).cooks_distance    # 쿡의 거리 값과 인덱스
# 쿡 값중 가장 큰 5개 관측치 확인
print(cd.sort_values(ascending = False))
# (인덱스) 번째에 해당하는 원본 자료 확인
print(advdf.iloc[[35,178,25,175,131]])
#         tv  radio  newspaper  sales
# 35   290.7    4.1        8.5   12.8
# 178  276.7    2.3       23.7   11.8
# 25   262.9    3.5       19.5   12.0
# 175  276.9   48.9       41.8   27.0
# 131  265.2    2.9       43.0   12.7
# 해석 : 대체적으로 tv 광고비는 높은데 그에 반해 sales가 작음(효율이 안나왓단 얘긴가봐)
#       - 모델이 예측하기 어려운 포인트들, 그니까 이걸 제낄 수 잇다면 어떨까 라는 얘기가 나왓어

# Cook's Distance 시각화
import statsmodels.api as sm
fig = sm.graphics.influence_plot(lmodel1, alpha = 0.05, criterion = 'cooks')     # alpha값은 유의수준이래
# plt.xlabel('leverage')  # 이게 절대값이래
# plt.ylabel('')
plt.show()
# 이 플롯에서 원의 크기가 큰 것들은 outlier일 가능성이 큰 데이터값들이래
# 그러면 일정 수치(데이터 수치가 너무 튄다 그니까 이정도 넘는 데이터는 제끼자)
# 를 넘는 데이터는 제낄 때 활용하나봐





#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.612
# Model:                            OLS   Adj. R-squared:                  0.610
# Method:                 Least Squares   F-statistic:                     312.1
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.47e-42    p값이야 이게
# Time:                        15:04:10   Log-Likelihood:                -519.05
# No. Observations:                 200   AIC:                             1042.
# Df Residuals:                     198   BIC:                             1049.
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
# tv             0.0475      0.003     17.668      0.000       0.042       0.053
# ==============================================================================
# Omnibus:                        0.531   Durbin-Watson:                   1.935
# Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
# Skew:                          -0.089   Prob(JB):                        0.716
# Kurtosis:                       2.779   Cond. No.                         338.
# ==============================================================================