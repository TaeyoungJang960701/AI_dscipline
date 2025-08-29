# 분류모델 성능 평가 관련
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

x, y = make_classification(n_samples = 100, n_features = 2, n_redundant=0, random_state = 123)
print(x[:3])
print(y[:3])

import matplotlib.pyplot as plt

# plt.scatter(x[:,0], x[:,1])
# plt.show()
# plt.close()

model = LogisticRegression().fit(x,y)
yhat = model.predict(x)
print('yhat : ', yhat[:3])

f_value = model.decision_function(x)    
# 결정 함수(판별 함수, 불확실성 추정 함수), 판별 경계선 설정을 위한 샘플 자료 얻기
print('f_value : ', f_value[:10])
df = pd.DataFrame(np.vstack([f_value, yhat, y]).T, columns = ['f','yhat','y'])
print(df.head())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,yhat)
print(cm)
acc = (cm[0][0] + cm[1][1]) / len(y)
recall = cm[0][0] / (cm[0][1] + cm[0][0])
precision = cm[0][0] / (cm[1][0] + cm[0][0])
specificity = cm[0][0] / (cm[1][0] + cm[0][0])
fallout = cm[1][0] / (cm[1][0] + cm[1][1])
print('acc(정확도) : ', acc)                    # (TP + TN) / 전체
print('recall(재현율) : ', recall)              # TP / (TP + FN)
print('precision(정밀도) : ', precision)        # TP / (TP + FP)
print('specificity(특이도) : ', specificity)    # TN / (TN + FP)
print('fallout(위양성률) : ', fallout)
print('fallout(위양성률) : ', 1 - specificity)
# acc(정확도) :  0.88
# recall(재현율) :  0.9166666666666666
# precision(정밀도) :  0.8461538461538461
# specificity(특이도) :  0.8461538461538461
# fallout(위양성률) :  0.15384615384615385
# fallout(위양성률) :  0.15384615384615385
# 정리하면 TPR은 1에 근사하면 좋고, FPR은 0에 근사하면 좋다
print('-' * 100)

from sklearn import metrics
ac_sco = metrics.accuracy_score(y,yhat)
print('ac_sco : ', ac_sco)      # ac_sco :  0.88
cl_rep = metrics.classification_report(y,yhat)
print(cl_rep)

#               precision    recall  f1-score   support

#            0       0.85      0.92      0.88        48
#            1       0.92      0.85      0.88        52

#     accuracy                           0.88       100
#    macro avg       0.88      0.88      0.88       100
# weighted avg       0.88      0.88      0.88       100

print('-' * 100)

fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ',fpr)
print('tpr : ', tpr)
print('분류결정임계값(thresholds) : ', thresholds)

# ROC 커브 시각화
# plt.plot(fpr, tpr, 'o-', label = 'Logistic Regression')
# plt.plot([0,1], [0,1], 'k--', label = 'random class(AUC 0.5)')
# plt.plot([fallout], [recall], 'ro', ms = 10)    # 위양성률과 재현율값 출력
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('ROC curve')
# plt.legend()
# plt.show()
# plt.close()

# AUC(Area Under the Cover) - ROC 커브의 면적
# 1에 가까울수록 
print('-' * 100)
print('AUC : ', metrics.auc(fpr, tpr))  # AUC :  0.9547275641025641

