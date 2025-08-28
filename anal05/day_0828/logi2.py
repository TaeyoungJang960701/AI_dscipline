# rain tomorow를 알아보자 내일 비가 올 확률을 지금까지의 데이터를 통해 알아보자

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print(data.head(),data.shape)
data2 = pd.DataFrame()
data2 = data.drop(['Date','RainToday'], axis = 1)
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
print(data2.head(),data2.shape)     # (366, 12)
print(data2.RainTomorrow.unique())  # [1 0]

# 학습데이터와 검정데이터로 분리
train, test = train_test_split(data2, test_size = 0.3, random_state = 42)
print(train.shape, test.shape)      # (256, 10) (110, 10)
print(data2.columns)
col_select = '+'.join(train.columns.difference(['RainTomorrow']))
print(col_select)
my_formula = 'RainTomorrow ~ ' + col_select

# model = smf.glm(formula = my_formula, data = train, family = sm.families.Binomial()).fit()
model = smf.logit(formula = my_formula, data = train).fit()

print(model.summary())
# print(model.params)

# Cloud랑 MaxTemp랑 WindSpeed를 빼서 나머지를 독립변수로 쓸거야
print('예측값 : ', np.rint(model.predict(test)[:5].values))
print('실제값 : ', test['RainTomorrow'][:5].values)

# 분류 정확도 확인
conf_tab = model.pred_table()
print('conf_tab : \n', conf_tab)    
print('분류 정확도 : ', (conf_tab[0,0] + conf_tab[1,1]) / len(train))
# pred_table 지원하지 않아 어떤 라이브러리를 써야될까
# 이거 모델을 로짓으로 해서 하니까 (glm 대신에) 컨퓨션 테이블이 잘 나와
# glm은 이 모듈을 지원을 안하나봐 별도의 라이브러리는 필요없엇어 

from sklearn.metrics import accuracy_score
pred = model.predict(test)
print('분류 정확도 : ', accuracy_score(test['RainTomorrow'], np.around(pred)))
# 분류 정확도 :  0.8727272727272727

















