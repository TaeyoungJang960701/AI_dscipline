# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) 
# Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.
# ols

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf
from scipy.stats import shapiro     # 이건 정규성 확인하기 위한 라이브러리
import seaborn as sns
import statsmodels.api as sm        # Quantile-Quantile plot 지원 (이게 큐큐 플랏)
from statsmodels.stats.diagnostic import linear_reset   # 모형 적합성 확인

plt.rc('font', family = 'Malgun Gothic')

df = pd.read_csv(
    'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv'
    )
print(df.head(2))
print(df.info())
df = df.drop([df.columns[6], df.columns[9], df.columns[10]], axis = 1)
print(df.corr())
print('-' * 100)

lmodel = smf.ols(formula = 'Sales ~ Income + Advertising + Price + Age', data = df).fit()
print('요약결과 : \n',lmodel.summary())
print('-' * 100)
# p값을 보면
# Income, Advertising, Price, Age < 0.05


# 작성된 모델 저장 후 읽어서 사용
# pickle 모듈을 사용할거야 근데 왜?
"""
import pickle

# 읽기

with open('mymodel.pickle', mode = 'wb') as obj:
    pickle.dump(lmodel, obj)

with open('mymodel.pickle', mode = 'rb') as obj:
    mymodel = pickle.dump(obj)
mymodel.predict('~~~')
"""

"""
# joblib 모듈 사용
import joblib
# 저장
joblib.dump(lmodel, 'mymodel.model')

# 읽기
mymodel = joblib.load('mymodel.model')
mymodel.predict('어쩌고저쩌고')
"""

# *** 선형회귀분석의 기본 충족 조건 ***
print(df.head())
print('-' * 100)
df_lm = df.iloc[:,[0,2,3,5,6]]
# 독립변수인 Income Advertising Price Age 만 하려햇는데 방법을 바꿧어

# 잔차항 얻기
fitted = lmodel.predict(df_lm)
residual = df_lm['Sales'] - fitted
print('잔차의 평균 : ', np.mean(residual))
print('-' * 100)

print('\n선형성 : 잔차가 일정하게 분포되어야 한다')

sns.regplot(x = fitted, y = residual, lowess = True, line_kws = {'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color = 'gray')
plt.show()

print('\n정규성 : 잔차항이 정규 분포를 따라야 함')
import scipy.stats as stats
sr = stats.zscore(residual)
(x,y), _ = stats.probplot(sr)
print(x,y)
sns.scatterplot(x = x, y = y)
plt.title('Scatterplot')
# sns.scatterplot(sr)       아니 이것도 되던데 이건 뭐야
plt.plot([-3,3], [-3,3], '--', color = 'gray')
plt.show()

print('-' * 100)
print('shapiro test : ', stats.shapiro(residual))
# ShapiroResult(
# statistic=np.float64(0.9949221268962878), 
# pvalue=np.float64(0.21270047355487404))       p-value > 0.05 따라서 정규성 만족이야

print('\n독립성 : 독립변수의 값이 서로 관련되지 않아야 한다')
# 더빈-왓슨값이 2에 아주 근사하다
# Durbin-Watson:  1.931     독립성은 검증됏어

import statsmodels.api as sm

print('Durbin-Watson : ', sm.stats.stattools.durbin_watson(residual))
# Durbin-Watson :  1.931498127082959
# ols의 summary()로 봐도 이 모듈로 봐도 똑같은 수가 나오긴 해 내부 모듈은 똑같다 이거지 

# 여기부턴 8/26일 수업으로 넘기자 내일 설명해주신대
print('\n등분산성 : ')

print('\n다중공선성 : ')