# MLP 실습 : 종양 데이터
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)

# 정규화하면 정확도가 더 올라간대 밑에는 정규화하려고 하는 코드 구문이다
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),\
                     solver = 'adam', learning_rate_init=0.1, verbose = 1)
# verbose 이건 이터레이션마다 반복 출력 할까1 안할까0 
mlp.fit(x_train, y_train)
pred = mlp.predict(x_test)

from sklearn.metrics import accuracy_score
print('실제값 : ', y_test[:5])
print('예측값 : ', pred[:5])
print('분류 정확도 : ', accuracy_score(y_test, pred))   # 분류 정확도 :  0.972027972027972
