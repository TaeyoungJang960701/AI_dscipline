# Validation Split 방식과 K-Fold 방식의 차이
# 메모장에 강의필기 메모장에 넣어놧어 Anal 06 - day_0915
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/diabetes.csv',
                  delimiter = ',', dtype = np.float32)
x = data[:, :-1]
y = data[:, -1]
print(x[:3])
print(y[:3])

def build_model():
    model = Sequential([
        Input(shape = (8,)),
        Dense(units = 64, activation = 'relu'),
        Dense(units = 32, activation = 'relu'),
        Dense(units = 1, activation = 'sigmoid')        
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# validation split 방식을 써보자
model_val = build_model()
history_val = model_val.fit(x, y, epochs = 50, batch_size = 32, validation_split = 0.2, verbose = 0)

val_acc = history_val.history['val_accuracy'][-1]

# KFold 방식 사용
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
kfold_accuracies = []

for train_idx, val_idx, in kf.split(x):
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_kf = build_model()
    model_kf.fit(x_train, y_train, epochs = 50, batch_size = 32, verbose = 0)
    
    y_pred = model_kf.predict(x_val)
    y_pred_label = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred_label)
    kfold_accuracies.append(acc)
    

# 비교 출력
print(f'[validation_split] 마지막 검증 정확도 : {val_acc:.4f}')
print(f'[KFold] 각 fold의 정확도 : {np.round(kfold_accuracies, 4)}')
print(f'[KFold] 평균 정확도 : {np.mean(kfold_accuracies):.4f}')

