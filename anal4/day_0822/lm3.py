# ML(기계학습 - 지도학습(supervised learning))
# 회귀분석(linear regression) : 입력 데이터에 대한 
# 잔차 제곱합이 최소가 되는 추세선(1차방정식 - 회귀선)을 만들고,
# 이를 통해 독립변수가 종속변수에 얼마나 영향을 주는지 인과관계를 분석
# 독립변수 : 연속형, 종속변수 : 연속형. 
# 두 변수는 상관관계가 있어야 하며, 인과관계를 보여야 한다.
# 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.datasets import make_regression # 연습용 데이터 생성 라이브러리래
import numpy as np

np.random.seed(12)

# 모델 생성 후 맛보기
# 방법 1 : make_regression을 그대로 사용 : model 생성은 안돼 그냥 확인용
# -----------------------------------------------------------------------------
x, y, coef = make_regression(n_samples = 50, n_features = 1, bias = 100, coef = True)     
# 샘플 개수 50개, 독립변수 1개, 초기치 100
print(x)
print('-' * 100)
print(y)
print('-' * 100)
print('coef : ',coef)   # coef :  89.47430739278907 이게 기울기값 w
# 회귀식 : y = wx + b => y = 89.47430739278907 * x + b
print('-' * 100)
# sklearn 은 2차원 배열을 만든대

y_pred = 89.47430739278907 * -1.70073563 + 100    # 당연히 숫자 집어넣는것보다 coef로 넣는게 더 정환한 값이 나오겟지
# -1.700 이건 임의생성된 x값의 첫번째값
print('y_pred : ',y_pred)
# y_pred값은 -52.17214255248879 인데 실제 데이터 y값은 -52.17214291 로 생각보다 믿을만하다

# 미지의 x에 대한 예측값 y 얻기
print('y_pred_new : ', 89.47430739278907 * 5 + 100)     # 한번 5 넣으면 뭐가 나올까 해서 넣은 수
# y_pred_new :  547.3715369639453   이 모델을 써서 차후 인풋값에 대한 아웃풋을 예상할 수 있어
print('-' * 100)
xx = x
yy = y

# -----------------------------------------------------------------------------
# 방법 2 : LinearRegression을 사용 - 모델 생성 된대
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# 여담인데 lm2 파일에서 만든 마이 리니어 리그레션 클래스 잇잖아 웬만하면 클래스로 정의해서 만들래
# func 정의해서 만드는건 조그마한것만 그렇게 하고 큰거는 클래스 정의로 하라는데
fit_model = model.fit(xx,yy)    # 학습 데이터로 모형 추정. 절편, 기울기 얻음

print(fit_model.coef_)      # 요게 기울기
print(fit_model.intercept_) # 요게 절편이래    알아본 방법은 API를 읽어보셧대 API 잘 알아볼 수 잇어야된대
# w = [89.47430739], b = 100.0 -----> y = 89.47430739 * x + 100.0

print(89.47430739 * xx[[0]] + 100.0) # [[-52.17214291]] 
# 여기서 xx[0] 이렇게 하지 않은 이유는 사이키 런은 2차원 배열을 받아야된대 그래서 어레이로 넘겨준거야
print('예측값y[0] : ', 89.47430739 * xx[[0]] + 100.0)   # [[-52.17214291]] 2차원으로 넘겨줫으니깐 [[]]
print('예측값y[0] : ', model.predict(xx[[0]]))          # [-52.17214291]

# 마지의 xx(5)에 대한 예측값 y얻기
print('미지의 인풋 x에 대한 예측값y[0] : ', model.predict(xx[[5]]))     # 이건 xx의 5번째 x를 불러온거지
print('미지의 인풋 x에 대한 예측값y[0] : ', model.predict([[5.5],[3.3]]))  # 이건 지피티가 고쳐준 코드
print('미지의 인풋 x에 대한 예측값y[0] : ', model.predict([[5],[3]])) # 이건 쌤이 스스로 고친 코드
# 원래 쌤 코드는 model.predict(xx[5.5],[3.3])) 이거엿거든 근데 이건 xx의 5.5번째잖아 말이 안된대 
# 지피티가 정수형만 쓰라 그러는데 의도대로 5.5와 3.3에 대한 예측값을 보려면 그냥 어레이로 입력하래 
# 그래서 model.predict([[5.5],[3.3]]))

# 쌤이 잠깐 어레이랑 인덱스랑 헷갈렷던 잠깐의 헤프닝
print('-' * 100)
# ------------------------------------------------------------------------
# 방법 3 : ols 사용 -> model 생성 O
import statsmodels.formula.api as smf   # 이 smf는 2차원을 원하지 않는대 그래서 차원을 1차원으로 다룰거야
import pandas as pd

x1 = xx.flatten()   # 2차원으로 저장된 xx를 1차원으로 떨어뜨려주는 구문
# x1 = xx.ravel()   이것도 차원 떨어뜨려주는 함수
print(xx, xx.shape)
print('-' * 100)

y1 = yy     # y는 원래부터 1차원이엇으니까 노상관

data = np.array([x1, y1])
print(data.shape)
print(data.T)
print(data.T.shape)
print('-' * 100)

df = pd.DataFrame(data.T)   # 보기 편할라고 트랜스포즈
df.columns = ['x1','y1']
print(df.head(2))
print('-' * 100)
model2 = smf.ols(formula = 'y1 ~ x1', data= df).fit() 
# 포뮬러 =  '종속변수 ~ 독립변수', 데이터는 df야, fit() 학습해

print(model2.summary())
print('-' * 100)
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     y1   R-squared:                       1.000
# Model:                            OLS   Adj. R-squared:                  1.000
# Method:                 Least Squares   F-statistic:                 1.905e+32
# Method:                 Least Squares   F-statistic:                 1.905e+32
# Date:                Fri, 22 Aug 2025   Prob (F-statistic):               0.00
# Time:                        11:59:45   Log-Likelihood:                 1460.6
# No. Observations:                  50   AIC:                            -2917.
# Df Residuals:                      48   BIC:                            -2913.
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    100.0000   7.33e-15   1.36e+16      0.000     100.000     100.000    # 여기서 절편값 100.0000
# x1            89.4743   6.48e-15   1.38e+16      0.000      89.474      89.474    # 여기서 기울기 알려줌
# x1            89.4743   6.48e-15   1.38e+16      0.000      89.474      89.474
# x1            89.4743   6.48e-15   1.38e+16      0.000      89.474      89.474
# ==============================================================================
# Omnibus:                        7.616   Durbin-Watson:                   1.798
# Prob(Omnibus):                  0.022   Jarque-Bera (JB):                8.746
# Skew:                           0.516   Prob(JB):                       0.0126
# Kurtosis:                       4.770   Cond. No.                         1.26
# ==============================================================================
# 위의 리니어 리그레션 클래스는 이 결과창을 보여주지 않아
# 이 ols.summary() 클래스 결과창이 담고잇는 정보가 엄청 많대 그래서 이게 좋대

print(x1[:2])   # [-1.70073563 -0.67794537]
new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]})     # 기존 자료로 예측값 확인
new_pred = model2.predict(new_df)
print('new_pred : ', new_pred.values)

# 전혀 새로운 독립변수로 종속변수 예측
new_df2 = pd.DataFrame({'x1':[123, -2.677]})
new_pred2 = model2.predict(new_df2)
print('new_pred2 : ', new_pred2.values)