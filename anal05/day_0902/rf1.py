# RandomForest 분류/예측 알고리즘
# 분류 알고리즘으로 titanic dataset 사용해 2진 분류
# Bagging 사용 : 데이터 샘플링(bootstrap)을 통해 모델을 학습시키고, 결과를 집계(Aggregation)
# 참고 : 우수한 성능을 원한다면 Boosting, 오버피팅이 걱정된다면 Bagging의 방식을 써라

# titanic dataset : feature - (pclass, age, sex, spouse, sibling), label (survived)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 한글 깨짐 방지, 음수 깨짐 방지
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')

print(df.head())
print(df.shape)     # (891, 12)
print(df.columns)
print(df.info())
print(df.isnull().any())

df = df.dropna(subset = ['Pclass','Age','Sex'])
print(df.shape)     # (714, 12)
# feature, label로 분리
df_x = df[['Pclass','Age','Sex']].copy()
print(df_x.head())      # Sex를 숫자화
encoder = LabelEncoder()
df_x.loc[:,'Sex'] = encoder.fit_transform(df_x['Sex'])
print(df_x.head())
df_y = df['Survived']
print(df_y.head())
print('-' * 100)
train_x, test_x, train_y,test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12)

model = RandomForestClassifier(criterion='entropy', n_estimators=500)
model.fit(train_x, train_y)
pred = model.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', np.array(test_y[:10]))
print('맞춘 갯수 : ', sum(test_y == pred))
print('전체 대비 맞춘 비율 : ', sum(test_y == pred) / len(test_y))
print('분류 정확도 : ', accuracy_score(test_y, pred))

# k-fold
cross_vali = cross_val_score(model, df_x, df_y, cv = 5)
print(cross_vali)
# [0.76223776 0.81118881 0.81818182 0.83216783 0.82394366]
print('교차검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5))  
# 교차검증 평균 정확도 :  0.80954
print('-' * 100)

# 중요변수 확인
print('특성(변수) 중요도 : ', model.feature_importances_)

import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align = 'center')
plt.xlabel('feature_importatnces_score')
plt.ylabel('features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()








