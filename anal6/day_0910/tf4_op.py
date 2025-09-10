# TensorFlow 연산자
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = tf.constant(3)

# 삼항연산 : condition
result1 = tf.cond(x > y, lambda:tf.add(x,y), lambda:tf.subtract(x,y))
print(result1)
# result = tf.cond(tf.greater(x,y), lambda: tf.add(x,y), lambda: tf.subtract(x,y))

# case 조건
f1 = lambda:tf.constant(1)
print(f1())
f2 = lambda:tf.constant(2)
a = tf.constant(5)
b = tf.constant(4)
result2 = tf.case([(tf.less(a,b), f1)], default = f2)
print(result2)


# 관계연산
print(tf.equal(1,2))
print(tf.not_equal(1,2))
print(tf.greater(1,2))
print(tf.greater_equal(1,2))
print(tf.less(1,2))

# 논리 연산
print(tf.logical_and(True, False))
print(tf.logical_or(True, False))
print(tf.logical_not(True))



# 유일 합집합
kbs = tf.constant([1,2,2,2,3])
val, idx = tf.unique(kbs)
print(val.numpy())
print(idx.numpy())

# reduce_~
ar = [[1,2],[3,4]]
print(tf.reduce_mean(ar).numpy())
print(tf.reduce_mean(ar, axis = 0).numpy())
print(tf.reduce_mean(ar, axis = 1).numpy())

# . . .
