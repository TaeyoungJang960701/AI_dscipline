# kaggle.com에서 Carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.
# ols 사용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import seaborn
import statsmodels.formula.api as smf

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv')
print(df.head(2))
print(df.info())
df = df.drop([df.columns[6],df.columns[9], df.columns[10]], axis= 1)
print(df.corr())
lmodel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=df).fit()
print('요약결과 : ', lmodel.summary())
# Income,Advertising,Price,Age 모두 < 0.05
# 작성된 모델 저장 후 읽어서 사용
"""
# pickle 모듈 사용
import pickle
# 저장
with open('mymodel.pickle', mode='wb') as obj:
    pickle.dump(lmodel, obj)
# 읽기
with open('mymodel.pickle', mode='rb') as obj:
    mymodel = pickle.load(obj)
mymodel.predict('~~~')
"""

# joblib 모듈 사용
import joblib
# 저장
#joblib.dump(lmodel, 'mymodel.model')
"""
# 읽기
mymodel = joblib.load('mymodel.model')
mymodel.predict('~~~')
"""

# *** 선형회귀분석의 기본 충족 조건 ***
print(df.head(3))
df_lm = df.iloc[:,[0,2,3,5,6]]
# 잔차항 얻기
fitted = lmodel.predict(df_lm)
residual = df_lm['Sales'] - fitted
print(residual[:3])
print('-----------------잔차의 평균 : ', np.mean(residual))

import seaborn as sns
print('\n선형성 : 잔차가 일정하게 분포되어야 함')
# 시각화로 확인
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='gray')
plt.show()   # 선형성 만족

print('\n정규성 : 잔차항이 정규 분포를 따라야 함')
import scipy.stats as stats
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x=x, y=y)
plt.plot([-3, 3], [-3, 3], '--', color='gray')
plt.show()
print()
print('shapiro test : ', stats.shapiro(residual))
# pvalue=0.212700473 > 0.05 이므로 정규성 만족

print('\n독립성 : 독립변수의 값이 서로 관련되지 않아야 한다')
# 듀빈-왓슨(Durbin-Watson) 검정으로 확인
# Durbin-Watson:1.931 -- 2에 근사하면 자기상관 없음
import statsmodels.api as sm
print('Durbin-Watson:', sm.stats.stattools.durbin_watson(residual))
# Durbin-Watson: 1.931498127082959  -- 2에 근사하면 자기상관 없음

print('\n등분산성(homoskedasticity, 等分散性) : ')
# . 등분산성 : 그룹 간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# 시각화
sr = stats.zscore(residual)
sns.regplot(x=fitted, y=np.sqrt(abs(sr)), lowess=True, line_kws={'color':'red'})
plt.show()

from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, lmodel.model.exog)
print(f'통계량 : {bp_test[0]:.4f}, p-value : {bp_test[1]:.4f}')
#  p-value : 0.8899 > 0.05 등분산성 만족

print('\n다중공선성 : ')
# 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.
# 분산팽창지수(VIF : Variance Inflation Factor) : 연속형의 경우 10을 넘으면 의심
from statsmodels.stats.outliers_influence import variance_inflation_factor
imsidf = df[['Income', 'Advertising', 'Price', 'Age']]
vifdf = pd.DataFrame()
vifdf['vif_value'] = \
    [variance_inflation_factor(imsidf.values, i) for i in range(imsidf.shape[1])]
print(vifdf)  # 모든 독립변수가 VIF 값 10 미만 이므로 다중공선성 문제 발생하지 않음

print()
# 저장된 모델 읽어 새로운 데이터에 대한 예측
import joblib
ourmodel = joblib.load('mymodel.model')
new_df = pd.DataFrame({'Income':[35,63,25],'Advertising':[6,3,11],'Price':[105,88,77],'Age':[33, 55, 22]})
new_pred = ourmodel.predict(new_df)
print('Sales 예측결과 :\n', new_pred)