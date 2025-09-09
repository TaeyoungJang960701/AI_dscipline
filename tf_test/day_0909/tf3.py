# constant / Variable 정리
import tensorflow as tf
import numpy as np

node1 = tf.constant(3, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)

imsi = tf.add(node1, node2)
print(imsi)

print('-' * 100)
node3 = tf.Variable(3, dtype = tf.float32)
node4 = tf.Variable(4.0)
tf.print(node3)
tf.print(node4)

imsi2 = tf.add(node3, node4)
print(imsi2)
node4.assign_add(node3)
print(node4)

print('-' * 100)
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a,b)
result = tf.cond(a < b, lambda:tf.add(10,c), lambda:tf.square(a))
print(result.numpy())

print('-' * 100)
# v = tf.Variable(1)
v = tf.Variable(2)

@tf.function    # autograph 기능에 의해 Graph 객체 환경에서 작업(코드는 필요에 의해서 자동변환됨) <- 이걸 C가 해준대
def find_nextFunc():
  v.assign(v + 1)
  if tf.equal(v % 2, 0):
    v.assign(v + 10)

find_nextFunc()
print(v.numpy())
print(type(find_nextFunc))
# 위의 오토그래프 주석하고 실행하면
# <class 'function'>
# 주석 떼고 실행하면
# <class 'tensorflow.python.eager.polymorphic_function.polymorphic_function.Function'>
# 오토그래프 함수 주고 실행하면 클래스가 바뀐대 그리고 오토그래프 줫을때 속도가 빠르대

print('1부터 3까지 합 출력 함수 작성')
def func1():
  imsi = tf.constant(0)     # imsi = 0과 동일
  su = 1
  for _ in range(3):
    # imsi = tf.add(imsi, su)
    # imsi = imsi + su
    imsi += su
  return imsi

kbs = func1()
print(kbs.numpy(),' ', np.array(kbs))

print('-' * 100)

imsi = tf.constant(0)

@tf.function
def func2():
  # imsi = tf.constant(0)
  global imsi

  su = 1
  for _ in range(3):
    imsi += su
  return imsi

mbc = func2()
print(mbc.numpy(), ' ', np.array(mbc))

print()
# imsi = tf.Variable(0)

def func3():
  imsi = tf.Variable(0)   # 상태를 가지는 객체(값이 동적이다)
  su = 1
  for _ in range(3):
    # imsi += su
    # imsi += su
    imsi.assign_add(su)

  return imsi

sbs = func3()
print(sbs.numpy(), ' ', np.array(sbs))

print('구구단 출력 -------------')

@tf.function
def gugu1(dan):
  su = tf.constant(0)
  for _ in range(9):
    su = tf.add(su, 1)
    # print(su)
    # print(su.numpy())
    # tf.print(su)
    # print('{} * {} = {:2}'.format(dan,su,dan*su))
    # tf.print('{} * {} = {:2}'.format(dan,su,dan*su))
    tf.print(dan, "*", su, "=", dan * su)
gugu1(3)


print()
@tf.function
def gugu2(dan):
  for i in range(1,10):
    result = tf.multiply(dan, i)        # 원소곱, tf.matmul() 이건 행렬곱 (내적 계산이 가능하대)
    tf.print(dan, "*", i, "=", result)
gugu2(5)