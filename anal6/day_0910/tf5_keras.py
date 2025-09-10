# keras 모듈로 논리회로 분류 모델 작성
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.models import load_model


# 1. 데이터 세트 생성 (입력은 매트릭스, 출력은 벡터)
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [1]])

# 2. 모델 구성
# model = Sequential([
#     Input(shape = (2,)),
#     Dense(units = 1),
#     Activation('sigmoid')
# ])
model = Sequential()
model.add(Input(shape = (2,)))
model.add(Dense(units = 1))
model.add(Activation('sigmoid'))    # 주석처리한 위 문단이랑 똑같은 말인데 이게 더 편해보여

# 3. 모델 학습 과정 설정
# Relu -> 0 이하의 수는 무조건 0으로 반환
model.compile(optimizer= 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent

# model.compile(optimizer= 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent
# model.compile(optimizer= 'adam', loss = 'binay_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent
# model.compile(optimizer= SGD(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent
# model.compile(optimizer= SGD(learning_rate = 0.01, momentum = 0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent
# model.compile(optimizer= RMSprop(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent
# model.compile(optimizer= Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])   # Stochastic Gradient Descent

# cost function(loss)의 최저값을 찾는 알고리즘
# binary_crossentropy 이항분

# 4. 모델 학습시키기
model.fit(x = x, y = y, epochs = 100, batch_size = 1, verbose = 0)

# 5. 모델 평가
loss_metrics = model.evaluate(x,y)
print('loss_metrics : ', loss_metrics)

# 6. 모델 사용하기 - 예측값 확인
proba = model.predict(x, verbose = 0)
print('proba : ', proba)
pred = (proba > 0.5).astype('int32')
print(pred.ravel())

# 모델 저장
model.save('test.keras')


# 모델 읽기
model2 = load_model('test.keras')

proba = model2.predict(x, verbose = 1)
print('proba : ', proba)
pred = (proba > 0.5).astype('int32')
print(pred.ravel())