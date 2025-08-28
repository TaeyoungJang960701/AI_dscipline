# LogisticRegression 클래스를 사용 : 다항분류
# 활성화 함수 사용(Activation Function)
# -> softmax 함수

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import 
import pickle
# from

iris = datasets.load_iris()
# print(iris.DESCR)

print(iris.keys())
print(iris.target)
x = iris['data'][:,[3]]     # Petal.Length
print(x)
y = (iris.target == 2).astype(np.int32)

# print(y[:3])
# print(type(y))

log_reg = LogisticRegression().fit(x,y)     # solver : lbfgs    이 솔버가 디폴트래
print(log_reg)

x_new= np.linspace(0,3,1000).reshape(-1,1)     # 새로운 예측값을 얻기 위해 독립변수 설정
print(x_new)

y_proba = log_reg.predict_proba(x_new)
# print(y_proba)

import matplotlib.pyplot as plt
plt.plot(x_new,y_proba[:,1],'r-', label = 'virginica')
plt.plot(x_new,y_proba[:,0],'b--', label = 'setosa')
# plt.xlabel('Petal Width')
# plt.legend()
# plt.show()
# plt.close()

