from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=66)

train_accuracy = []
test_accuracy = []
neighbors_set = range(1, 11, 2)

for n_neighbors in neighbors_set:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors, p = 1, metric = 'minkowski')
    clf.fit(x_train, y_train)
    train_accuracy.append(clf.score(x_train,y_train))
    test_accuracy.append(clf.score(x_test,y_test))
    
import numpy as np
print('train 분류 정확도 평균 : ', np.mean(train_accuracy))
print('test 분류 정확도 평균 : ', np.mean(test_accuracy))

plt.plot(neighbors_set, train_accuracy, label = 'train_accuracy')
plt.plot(neighbors_set, test_accuracy, label = 'test_accuracy')
plt.ylabel('acc')
plt.xlabel('k')
plt.legend()
plt.show()
plt.close()