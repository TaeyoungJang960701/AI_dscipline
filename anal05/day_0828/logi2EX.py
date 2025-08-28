# [로지스틱 분류분석 문제1]

# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 
# 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

import pandas as pd
from io import StringIO     # 일일이 치지 않고 데이터프레임 넣으려고 쓴 라이브러리
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score
import numpy as np

data_str = """
요일,외식유무,소득수준
토,0,57
토,0,39
토,0,28
화,1,60
토,0,31
월,1,42
토,1,54
토,1,65
토,0,45
토,0,37
토,1,98
토,1,60
토,0,41
토,1,52
일,1,75
월,1,45
화,0,46
수,0,39
목,1,70
금,1,44
토,1,74
토,1,65
토,0,46
토,0,39
일,1,60
토,1,44
일,0,30
토,0,34
"""

# 이건 독립변수가 소득수준이고 외식유무가 종속변수다

df = pd.read_csv(StringIO(data_str))
# print(df.head())
# print(df.shape,df.외식유무.unique())     # (28, 3) [0 1]


train, test = train_test_split(df, test_size = 0.3, random_state = 42)
print(train.shape, test.shape)            # (19, 3) (9, 3)

x_train = train['소득수준']
y_train = train['외식유무']

x_test = test['소득수준']
y_test = test['외식유무']

model = smf.logit(formula = '외식유무 ~ 소득수준', data = train).fit()
print(model.summary())
#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:                   외식유무   No. Observations:                   28
# Model:                          Logit   Df Residuals:                       26
# Method:                           MLE   Df Model:                            1
# Date:                Thu, 28 Aug 2025   Pseudo R-squ.:                  0.4796
# Time:                        13:01:54   Log-Likelihood:                -10.062
# converged:                       True   LL-Null:                       -19.337
# Covariance Type:            nonrobust   LLR p-value:                 1.656e-05
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     -9.4482      3.588     -2.633      0.008     -16.481      -2.415
# 소득수준           0.2022      0.078      2.587      0.010       0.049       0.355
# ==============================================================================

# 예측값과 실제값을 비교해보자
print('실제값 : \n', np.rint(model.predict(test)[:5].values))
print('예측값 : \n', test['외식유무'][:5].values)

conf_tab = model.pred_table()   # train 기준 컨퓨전 매트릭스
print('conf_tab : \n', conf_tab)
print('컨퓨전 테이블을 이용한 train 데이터 기준 분류 정확도 : \n', (conf_tab[0,0] + conf_tab[1,1]) / len(train))
#  0.8947368421052632

# accuracy score를 사용한 분류 정확도를 알아보자
pred = model.predict(test)
print('accuracy score를 이용한 test 데이터 기준 분류 정확도 : \n', accuracy_score(test['외식유무'],np.around(pred)))
# 0.6666666666666666