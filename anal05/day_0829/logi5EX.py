# [로지스틱 분류분석 문제2] 
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값0(착용X), 값1(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/bodycheck.csv')
print(df.head())
x = df[['게임','TV시청']]
y = df[['안경유무']]
# print(x)
# print(y)
df = df.drop(['번호','신장','체중'], axis = 1)
print(df)
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)
print('데이터 모양 확인 : ', x_train.shape, x_test.shape,y_train.shape,y_test.shape)
# print('x_train : \n',x_train)
# print('x_test : \n',x_test)
# print('y_train : \n',y_train)
# print('y_test : \n',y_test)

train_df = pd.concat([x_train, y_train], axis = 1)
model = smf.logit(formula = '안경유무 ~ 게임 + TV시청', data = train_df).fit()
print(model.summary())

print('예측값 : ', np.rint(model.predict(x_test)[:5].values))
print('실제값 : ', y_test['안경유무'][:5].values)

conf_tab = model.pred_table()
print('conf_tab : \n', conf_tab)
print('confusion table 분류 정확도 : ', (conf_tab[0,0] + conf_tab[1,1]) / conf_tab.sum())

y_pred = np.rint(model.predict(x_test))
print('accuracy score', accuracy_score(y_test,y_pred))












# [로지스틱 분류분석 문제3]
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 얘를 사용해도 됨   'testdata/advertisement.csv' 
# 참여 칼럼 : 
#    - Daily Time Spent on Site : 사이트 이용 시간 (분)
#    - Age : 나이,
#    - Area Income : 지역 소득,
#    - Daily Internet Usage :일별 인터넷 사용량(분),
#    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
# 모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
# 새로운 데이터로 분류 작업을 진행해 본다.