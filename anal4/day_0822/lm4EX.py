# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 
# 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 
# 운동량에 대한 데이터는 아래와 같다.

#  - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 
# 회귀분석 모델을 작성한 후에 예측하시오.

#  - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 
# 회귀분석 모델을 작성한 후에 예측하시오.

#     참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

# 구분,지상파,종편,운동
# 1,0.9,0.7,4.2
# 2,1.2,1.0,3.8
# 3,1.2,1.3,3.5
# 4,1.9,2.0,4.0
# 5,3.3,3.9,2.5
# 6,4.1,3.9,2.0
# 7,5.8,4.1,1.3
# 8,2.8,2.1,2.4
# 9,3.8,3.1,1.3
# 10,4.8,3.1,35.0
# 11,NaN,3.5,4.0
# 12,0.9,0.7,4.2
# 13,3.0,2.0,1.8
# 14,2.2,1.5,3.5
# 15,2.0,2.0,3.5

# 먼저 이상치는 다 지우고 평균값 메겨야겟다
# 그다음 관계성 유의미성 해보고
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import StringIO     # 이건 저 데이터들 하나하나 손으로 안치고 저장하려고 넣은거 지피티가 알려줌
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf   # 이건 ols 쓰려고 넣은 라이브러리

data_str = """
1,0.9,0.7,4.2
2,1.2,1.0,3.8
3,1.2,1.3,3.5
4,1.9,2.0,4.0
5,3.3,3.9,2.5
6,4.1,3.9,2.0
7,5.8,4.1,1.3
8,2.8,2.1,2.4
9,3.8,3.1,1.3
10,4.8,3.1,35.0
11,NaN,3.5,4.0
12,0.9,0.7,4.2
13,3.0,2.0,1.8
14,2.2,1.5,3.5
15,2.0,2.0,3.5
"""
df = pd.read_csv(StringIO(data_str), header = None, names = ['구분','지상파','종편','운동'] ) # 구분,지상파,종편,운동
print(df)
data = df.dropna(how = 'any', axis = 0)
mean_tv = np.mean(data['지상파'])
print(mean_tv)
# 여기까지 지상파 열의 평균 구하는거

df['지상파'] = df['지상파'].fillna(mean_tv)
print(df)

df = df.drop(df[df['운동'] >= 10].index, axis = 0)
print(df)
# 여기까지 전처리 끝

x1 = df.지상파
y1 = df.운동
print(x1,'\n',y1)
print('-' * 100)

print(np.corrcoef(x1,y1)[0,1])    # 알값이 -0.8655346605559782 절대값이 1에 가까워 아주 유의미하다
model = stats.linregress(x1,y1)
print(model)
# plt.scatter(x1,y1)
# plt.plot(x1,model.slope * x1 + model.intercept)
# plt.show()
print('-' * 100)

# LinregressResult(slope=np.float64(-0.6684550167105406),   # w값은 대충 이거
# intercept=np.float64(4.709676019780582),                  # 절편값 대충 4.71
# rvalue=np.float64(-0.8655346605559783),                   # 결정계수 절대값 1에 근사
# pvalue=np.float64(6.347578533142469e-05),                 # p-value는 0.05 이하 즉, 둘은 상관관계가 잇어보여
# stderr=np.float64(0.11166162336231236), 
# intercept_stderr=np.float64(0.3226596187355018))

print(df.지상파)
print('티비 시청시간에 따른 운동량의 변화 예측 : ',
      np.polyval([model.slope, model.intercept], np.array([4]))
      )
# 예측 모델로 본, 4시간의 지상파 시청시간에 따른 운동량 [2.03585595]이 
# 실제 입력 데이터와 많이 비슷해 보이기는 하다


# 이제 지상파와 종편의 관계를 알아보자
# 또 유의미성 알아볼라면 r값 가져오고 피밸류 보고 판단해야돼
# 알값의 절대값이 1에 가까울수록 데이터들끼리의 상관관계가 더 커지고
# 피밸류가 0.05보다 작다면 둘 사이 관계가 독립적이라는 귀무가설이 기각되고
# 두 데이터는 종속적이라는 대립가설을 채택하게 돼
# 근데 선후관계는 피밸류를 먼저 보는게 맞는것같애 보고나서 
# 종속적이면 얼마나 종속적인지를 보는게 알값인가봐

x2 = df.지상파
y2 = df.종편

print('지상파 시청시간과 종편 시청시간의 상관관계 정도 R값 : ',np.corrcoef(x2,y2)[0,1])
# 0.8875299693193012        양의 1에 가깝다 관계가 커보여

# 이건 ols 서머리 한번 써보자
print(x2.shape,y2.shape)    # (14,) (14,) 이게 뭔소리여
# print(x2,y2)
# x2 = x2.flatten()     
# 이거 안해도 된대 (14,) 이건 14x1의 1차원의 데이터래 그래서 오류나더라 이미 플랫하다고

print(x2, x2.shape)
print('-' * 100)
data = np.array([x2, y2])
model2 = smf.ols(formula = '종편 ~ 지상파', data = df).fit()

print(model2.summary())
print('-' * 100)

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     y2   R-squared:                       0.788
# Model:                            OLS   Adj. R-squared:                  0.770
# Method:                 Least Squares   F-statistic:                     44.53
# Date:                Fri, 22 Aug 2025   Prob (F-statistic):           2.28e-05
# Time:                        15:41:36   Log-Likelihood:                -11.223
# No. Observations:                  14   AIC:                             26.45
# Df Residuals:                      12   BIC:                             27.72
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.2952      0.335      0.882      0.395      -0.434       1.024
# x2             0.7727      0.116      6.673      0.000       0.520       1.025
# ==============================================================================
# Omnibus:                        2.821   Durbin-Watson:                   1.628
# Prob(Omnibus):                  0.244   Jarque-Bera (JB):                1.752
# Skew:                           0.857   Prob(JB):                        0.416
# Kurtosis:                       2.749   Cond. No.                         6.81
# ==============================================================================
# 결과표 보면 coef칸에 intercept는 절편, 지상파값은 기울기값

print(x2[:2])
print('-' * 100)
new_df = pd.DataFrame({'지상파':[4]})   # 지상파 시간 4시간에 대한 종편 시청시간 예상시간
pred = model2.predict(new_df)
print(pred)
# 지상파 4시간 봣으면 종편은 아마 3.385911 시간을 봣을거래
print('-' * 100)

plt.scatter(x2,y2)
plt.plot(x2,model.slope * x2 + model.intercept)
plt.show()