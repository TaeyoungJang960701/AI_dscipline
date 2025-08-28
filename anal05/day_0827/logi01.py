'''
## Logistic Regression
- 독립변수 feature,x : 연속형, 종속변수 label,y : 범주형
- 이항 분류(다항 분류도 가능하다)
- 출력된 연속형 자료(확률)를 logit 변환해서 최종적으로 sigmoid function에 의해 0 ~ 1 사이의 실수값이 나온다.
    - 0.5를 기준으로 0과 1 로 분류한다.
'''
# sigmoid function 맛보기
import math
def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

# 0 ~ 1 사이 값이 출력되는 것을 볼 수 있다.
'''
print(sigmoidFunc(3))
print(sigmoidFunc(1))
print(sigmoidFunc(-123))
print(sigmoidFunc(0.123))
'''

# mtcar dataset으로 실습
import statsmodels.api as sm
mtcardata = sm.datasets.get_rdataset('mtcars')

# 데이터 확인
print(mtcardata.keys()) # dict_keys(['data', '__doc__', 'package', 'title', 'from_cache', 'raw_data'])
mtcars = mtcardata.data
print(mtcars.head(2))

# mpg(연비), hp(마력)가 am(자동/수동)에 영향을 준다고 가정한다.
mtcar = mtcars.loc[:,['mpg','hp','am']]
print(mtcar.head(3))
print(mtcar['am'].unique()) # [1 0] 1 - 수동, 0 - 자동

# 연비와 마력에 따른 변속기 분류 모델 작성(수동 or 자동)
# 모델 작성 방법 1 : logit()
import statsmodels.formula.api as smf

formula = 'am ~ hp + mpg' # 종속변수 ~ 독립변수
model1 = smf.logit(formula=formula, data=mtcar).fit() # 생김새 ols와 비슷하게 생겼다.
print(model1) # 객체 출력됨
print(model1.summary()) # Logit Regression 결과표 출력
'''
- 설명력 없음
                           Logit Regression Results
==============================================================================
Dep. Variable:                     am   No. Observations:                   32
Model:                          Logit   Df Residuals:                       29
Method:                           MLE   Df Model:                            2
Date:                Wed, 27 Aug 2025   Pseudo R-squ.:                  0.5551
Time:                        17:08:47   Log-Likelihood:                -9.6163
converged:                       True   LL-Null:                       -21.615
Covariance Type:            nonrobust   LLR p-value:                 6.153e-06
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
hp             0.0550      0.027      2.045      0.041       0.002       0.108
mpg            1.2596      0.567      2.220      0.026       0.147       2.372
==============================================================================
- P>|z|가 0.05보다 작으므로 hp, mpg 독립변수로 유의하다.
- 모델이 유의한지는 알려주지 않음
'''

# 모델이 만들어졌으므로 예측값과 실제값 비교
import numpy as np
# print(f'예측값 : {model1.predict()}') # 0~1 사이의 값이 출력된다.
pred = model1.predict(mtcar[:10])
print(f'예측값 : {pred.values}')
print(f'예측값 : {np.around(pred.values)}') # [0 0 1 0 0 0 0 1 1 0]
print(f'실제값 : {mtcar['am'][:10].values}') # [1 1 1 0 0 0 0 0 0 0]
print()
# 분류 모델의 정확도(accuracy) 확인
conf_tab = model1.pred_table() # 수치에 대한 집계표
print(f'confusion matrix :\n{conf_tab}')
# [[16.(모델이 맞다고해서 맞춘 개수)  3.] 1종 오류?
#  [ 3. 10.(모델이 틀렸다고해서 맞춘 개수)]] 2종 오류?

print(f'분류 정확도 : {(16+10)/len(mtcar)}') # 모델이 맞춘 개수 = 맞은 개수 / 전체 개수
print(f'분류 정확도 : {(conf_tab[0][0]+conf_tab[1][1])/len(mtcar)}')

from sklearn.metrics import accuracy_score
pred2 = model1.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))
print('-' * 100)

# 모델 작성방법 2 : glm(generalied linear model)
model2 = smf.glm(formula = formula, data = mtcar, family = sm.families.Binomial()).fit()
print(model2.summary())
print(model2)
glm_pred = model2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glm_pred.values))
print('glm 실제값 : ', mtcar['am'][:10].values)

print('glm 분류 정확도 : ', accuracy_score(mtcar['am'],np.around(pred2)))

print('\n새로운 값(hp, mpg)으로 변속기(am) 분류 예측')
newdf = mtcar.iloc[:2].copy()
# print(newdf)

newdf['mpg'] = [10, 30]
newdf['hp'] = [120, 90]
print(newdf)

new_pred = model2.predict(newdf)
print('새로운 값(hp,mpg)에 대한 변속기', np.round(new_pred.values))

import pandas as pd
newdf2 = pd.DataFrame({'mpg' : [10, 30, 50, 5], 'hp' : [80,110, 130, 50]})
new_pred2 = model2.predict(newdf2)
print('new_pred2 : ', new_pred2.values)
print('new_pred2 : ', np.around(new_pred2.values))
print('new_pred2 : ', np.rint(new_pred2.values))

