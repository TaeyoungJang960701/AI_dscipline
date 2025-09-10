# tf에서 변수 선언 후 사용
import tensorflow as tf
print(tf.__version__)

f = tf.Variable(1.0)
v = tf.Variable(tf.ones((2,)))
m = tf.Variable(tf.ones((2,1)))
print(f)
print(v)
print(m)

tf.print(m)
print('-' * 100)
v1 = tf.Variable(1)
v1.assign(10)   # 변수에 값 할당
print(v1)

v2 = tf.Variable(tf.ones(shape = (1, 2)))
v2.assign([[20, 20]])
print(v2)

v3 = tf.Variable(tf.ones(shape = (1, 2)))
v3.assign([[30, 40]])
print(v3)

v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1 * v2 + 10
print(v3)
print()
var = tf.Variable([1,2,3,4,5], dtype = tf.float32)
result = var + 10
print(result)

print('-' * 100)
w = tf.Variable(tf.ones(shape = (1,)))
b = tf.Variable(tf.ones(shape = (1,)))
w.assign([2])
b.assign([3])

def func1(x):
  return w * x + b

out_a1 = func1(3)
print('out_a1 : ',out_a1)
print(type(func1))

print()
@tf.function    # auto graph 기능
def func2(x):
  return w * x + b

out_a2 = func2(3)
print('out_a2 : ',out_a2)
print(type(func2))

# 난수
rand = tf.random.uniform([1], 0, 1)
print(rand)
rand2 = tf.random.normal([4], 0, 1)
print(rand2)

aa = tf.ones((2,1))
print(aa.numpy())
m = tf.Variable(tf.zeros((2,1)))
print(m.numpy())
m.assign(aa)        # 치환
print(m.numpy())
m.assign_add(aa)    # 더하기 후 치환
print(m.numpy())
m.assign_sub(aa)    # 더하기 후 치환
print(m.numpy())

