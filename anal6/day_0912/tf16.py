import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import tensorflow as tf

train_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true', sheet_name = 'train')
test_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true', sheet_name='test')
# train_df = pd.read_excel('https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx')
# 주석의 코드 뒤에 ?raw=true랑 sheet 이름까지 넣엇어

print(train_df.head())
print(test_df.head())
print('-' * 100)

x_train = train_df.drop(['가격'], axis = 1)     # train의 feature는 가격을 뺀 나머지 데이터로 하자
x_test = test_df.drop(['가격'], axis = 1)       # test의 feature
y_train = train_df[['가격']]                    # train의 label
y_test = test_df[['가격']]                      # test의 label

print(x_train.head(2))
print(x_test.head(2))
print('-' * 100)
print(y_train.head(2))
print(y_test.head(2))
print('-' * 100)

print(x_train.columns)
print(x_train.shape)
print('-' * 100)

print(set(x_train.종류))            # {'대형', '중형', '준중형', '소형'}
print(set(x_train.연료))            # {'LPG', '가솔린', '디젤'}
print(set(x_train.변속기))          # {'자동', '수동'}
print('-' * 100)

# 종류, 연료, 변속기 열에 대해서는 LabelEncoder(), OneHotEncoder()를 적용
transformer = make_column_transformer((OneHotEncoder(), ['종류','연료','변속기']), remainder = 'passthrough')
# 위의 아규먼트에서     remainder = . . .의 디폴트값은
# remainder = 'drop'이래 (remainder 옵션이 나머지 열을 떨어뜨릴지(drop), 그대로 둘지(passthrough) 결정함.)
# 열이 transformer에 전달된대 근데 이건 지피티 돌려보자 못알아듣겟다

transformer.fit(x_train)
x_train = transformer.transform(x_train)        # 3개의 컬럼을 포함해 모든 컬럼이 표준화됨
x_test = transformer.transform(x_test)
print(x_train[:3], x_train.shape)               # (71, 16)
print(y_train[:3], y_train.shape)               # (71, 1)