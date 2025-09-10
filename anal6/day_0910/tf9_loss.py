# 텐서플로우는 자동 미분(주어진 입력 변수에 대한 연산의 gradient를 계산하는 것)을 위한 tf.GradientGape Api를 제공
import tensorflow as tf
import numpy as np

x = tf.Variable(5.0)
w = tf.Variable(0.0)

@tf.function
def train_step():
  # GradientTape : 연산 과정을 기억해뒀다가 나중에 자동으로 미분(gradient)을 계산해주는 클래스
  with tf.GradientTape() as tape:
    y = tf.multiply(w, x)   # b는 0으로 간주하는거지
    loss = tf.square(tf.subtract(y, 50))
  grad = tape.gradient(loss, w)     # 자동 미분
  mu = 0.01   # learning rate(학습률)
  w.assign_sub(mu * grad)
  return loss

for i in range(10):
  loss = train_step()
  print('{:1}, w : {:.5f}, loss : {:.5f}'.format(i, w.numpy(), loss.numpy()))


# keras.optimizers 패키지에 있는 Adam, SGD, RMSprop . . . 사용
opti = tf.keras.optimizers.SGD(learning_rate= 0.01)
x = tf.Variable(5.0)
w = tf.Variable(0.0)

@tf.function
def train_step2():
  # GradientTape : 연산 과정을 기억해뒀다가 나중에 자동으로 미분(gradient)을 계산해주는 클래스
  with tf.GradientTape() as tape:
    y = tf.multiply(w, x)   # b는 0으로 간주하는거지
    loss = tf.square(tf.subtract(y, 50))
  grad = tape.gradient(loss, w)     # 자동 미분

  opti.apply_gradients([(grad, w)])

  return loss

for i in range(10):
  loss = train_step2()
  print('{:1}, w : {:.5f}, loss : {:.5f}'.format(i, w.numpy(), loss.numpy()))

# 선형회귀 모형 작성
# keras.optimizers 패키지에 있는 Adam, SGD, RMSprop . . . 사용
opti = tf.keras.optimizers.SGD(learning_rate= 0.01)

tf.random.set_seed(2)
w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))

@tf.function
def train_step3(x, y):
  # GradientTape : 연산 과정을 기억해뒀다가 나중에 자동으로 미분(gradient)을 계산해주는 클래스
  with tf.GradientTape() as tape:
    hypo = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
  grad = tape.gradient(loss, [w,b])     # 자동 미분- loss를 w와 b로 미분하시오

  opti.apply_gradients(zip(grad, [w, b]))
  return loss

x = [1.,2.,3.,4.,5.]            # 이게 feature
y = [1.2, 2.0, 3.0, 3.5, 5.5]   # 이건 label

w_vals = []
cost_vals = []

for i in range(1, 101):
  cost_val = train_step3(x, y)
  cost_vals.append(cost_val.numpy())
  w_vals.append(w.numpy())
  if i % 10 == 0:
    print(cost_val)

print(cost_vals)
print(w_vals)

import matplotlib.pyplot as plt
plt.plot(w_vals, cost_vals, 'o--', )
plt.xlabel('w')
plt.ylabel('cost')
plt.show()
plt.close()

print('cost가 최소값일 때 w : ', w.numpy())
print('cost가 최소값일 때 b : ', b.numpy())

y_pred = tf.add(tf.multiply(x, w), b)
print('y_pred : ', y_pred)

plt.plot(x, y, 'ro', label = 'real')
plt.plot(x,y_pred, 'b-', label = 'real')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
plt.close()

# 새 값으로 예측하기
new_x = [3.5, 9.0]
new_pred = tf.multiply(new_x,w) + b
print('예측 결과 : ', new_pred.numpy())
