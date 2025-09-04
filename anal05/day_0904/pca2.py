# 치원 축소(PCA - 주성분 분석) : 

import numpy as np
import pandas as pd

# 독립변수(feature)
x1 = [95,91,66,94,68]
x2 = [56,27,25,1,98]
x3 = [57,34,9,79,4]

x = np.stack((x1,x2,x3), axis = 0)
print(x)

x = pd.DataFrame(x.T, columns = ['x1','x2','x3'])
print(x)

print('표준화 처리')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
print(x_std)    # 표준화한 값
print(scaler.inverse_transform)     # 표준화를 원복

print('PCA 처리')
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
print(pca.fit_transform(x_std))
print(pca.inverse_transform(pca.fit_transform(x_std)))
print(scaler.inverse_transform(pca.inverse_transform(pca.fit_transform(x_std))))

print('와인 데이터로 분류(radomfoerst)')
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split

datas = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv')
print(datas.head())

x = np.array(datas.iloc[:,0:12])
y = np.array(datas.iloc[:,12])
print(x[:2])
print(y[:2], set(y))
train_x, test_x,train_y, test_y = train_test_split(x,y, test_size = 0.3, random_state = 1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
model = RandomForestClassifier(criterion = 'entropy', n_estimators=100).fit(train_x,train_y)
pred = model.predict(test_x)
print('pred : ', pred[:5])
print('acc : ', sklearn.metrics.accuracy_score(test_y,pred))    # acc :  0.9948691636736788
print('-' * 100)

pca = PCA(n_components=3)   # 연습이니까 3개만 한대 근데 원래 만이 해야되나봐
x_pca = pca.fit_transform(x)
print(x[:3])
print(x_pca[:3])

train_x, test_x,train_y, test_y = train_test_split(x_pca,y, test_size = 0.3, random_state = 1)
model2 = RandomForestClassifier(criterion = 'entropy', n_estimators=100).fit(train_x,train_y)
pred2 = model2.predict(test_x)
print('pred2 : ', pred2[:5])
print('acc2 : ', sklearn.metrics.accuracy_score(test_y,pred2))    # acc :  0.952796305797845
print('-' * 100)
