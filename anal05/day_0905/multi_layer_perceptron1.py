# 단층 신경망(뉴런,노드) - Perceptron 
# : input의 가중치합에 임계값을 기준으로 2가지 output 중 하나를 출력하는 간단한 구조

# 다층 신경망으로 논리회로 분류
# MLP

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# label = np.array([0,0,0,1])       # and 연산 준비
# # and 연산은 퍼셉트론이 충분히 예측해낸대
# label = np.array([0,1,1,1])       # or 연산
label = np.array([0,1,1,0])         # xor 연산

ml = MLPClassifier(hidden_layer_sizes=30,\
                    solver = 'adam', learning_rate_init=0.01).fit(feature, label)
# ml = MLPClassifier(hidden_layer_sizes=(10,10,10),\
#                     solver = 'adam', learning_rate_init=0.01).fit(feature, label)
# hidden_layer_sizes = (10,10,10) 으로도 선언할 수 있대

# learning rate을 너무 촘촘하게 해버리면 학습 속도가 과하게 느려지지만 정확한 그래프 추적이 가능하고,
# 숫자를 키워서 듬성듬성 해버리면 속도는 빠르지만 본 그래프에 대한 정확한 추적이 거의 불가

print(ml)
pred = ml.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))

# 지금 맥스 이터, 그니까 학습 횟수를 1000번 줫는데 학습 중단 기준이 잇대
# 최적의 학습을 햇다고 판단하면 더이상 학습하려 하지 않는다네
# 내부 알고리즘으로 그렇게 돼잇다더라