import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

# 예시 데이터 (실제값, 예측값)
y_true = np.array([1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1, 0, 0])

# 혼동 행렬
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# 잘못 분류된 인덱스 확인
import numpy as np
misclassified_idx = np.where(y_true != y_pred)[0]
print("잘못 분류된 인덱스:", misclassified_idx)