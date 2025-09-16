# 다항분류 : 출력값이 softmax 함수로 인해 여러 개의 확률값으로 출력.
# 이때 확률값이 가장 큰 인덱스를 분류의 결과로 얻음

# softmax function을 작성해보자
import numpy as np


"""
def softmaxFunc(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1.0, 1.2, 1.5])
result = softmaxFunc(a)
print(result)
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical   # one-hot encoding 지원
import matplotlib.pyplot as plt

np.random.seed(1)

# data 준비
xdata = np.random.random((1000,12))     # 시험점수라고 가정
ydata = np.random.randint(5, size = (1000,1))

print(xdata[:5])        # feature
print(ydata[:5])        # label
# 정수를 다섯가지 형태로 출력될 수 있도록 모양 변경을 하는 것이다
# 원핫 처리

ydata = to_categorical(ydata, num_classes = 5)
# ydata를 5개의 원핫 처리할게
print(ydata[:5])
# print([int(np.argmax(i)) for i in ydata[:2]])      # 원핫인코딩 값을 원복

# model
model = Sequential()
model.add(Input(shape = (12,)))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 5, activation = 'softmax'))
print(model.summary())

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model.compile(optimizer = '', loss = 'categorical_crossentropy', metrics = ['accuracy'])


print('learning rate : ', model.optimizer.learning_rate.numpy())

history = model.fit(xdata, ydata, epochs = 1000, batch_size = 32, verbose = 0)

model_eval = model.evaluate(xdata, ydata)
print('모델 평가 결과 : ', model_eval)

# 시각화
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 4))

ax1.plot(history.history['loss'])
ax1.set_title('Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')

ax2.plot(history.history['accuracy'])
ax2.set_title('Loss')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
plt.show()

# 분류 예측 결과 보기
np.set_printoptions(suppress = True)
np.set_printoptions(precision = 5)
print('예측값 : \n', model.predict(xdata[:5]))
print('예측값 : \n', np.argmax(model.predict(xdata[:5]), axis = 1))
print('실제값 : \n', ydata[:5])
print('실제값 : \n', [int(i + 1) for i in np.argmax(ydata[:5], axis = 1)])

print('-' * 100)
# 새로운 값으로 예측
x_new = np.random.random([1,12])
print(x_new)
new_pred = model.predict(x_new)
print('분류 결과 : ', new_pred, ', 모두 더하면 : ', np.sum(new_pred))
print('분류 결과 : ', np.argmax(new_pred))

# 레이블에 해당하는 과목명 출력
classes = np.array(['국어', '영어','수학','과학','체육'])
print('예측값 : ', classes[np.argmax(new_pred, axis = 1)])

