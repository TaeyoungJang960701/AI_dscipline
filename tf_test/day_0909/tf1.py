import tensorflow as tf
print(tf.__version__)

import tensorflow as tf

print('즉시 실행 모드 : ', tf.executing_eagerly())

# Tensor 생성
print(1, type(1))  # 파이선 상수
print(tf.constant(1), type(tf.constant(1)))     # 0-dimension Tensor -> 스칼라
print(tf.constant([1]), type(tf.constant([1])))     # 1-dimension Tensor -> 1차원 배열이라고만 생각하자
print(tf.constant([[1]]), type(tf.constant([[1]])))     # 2-dimension Tensor -> 1차원 배열이라 생각하자

print()
a = tf.constant([1,2])
b = tf.constant([3,4])
c = a + b
print(c)
d = tf.constant([3])
e = c + d
print(e)
# 1차원 선언한 d의 3값이 c의 각 요소에 3씩 둘다 더해진거 보이지 이게 브로드캐스팅

f = tf.add(c,d)
print(f)
# 결과는 똑같애 아래 결과로 똑같지만 그 내부는 다르다네
# tf.Tensor([7 9], shape=(2,), dtype=int32)
print('-' * 100)


print(7)
print(tf.convert_to_tensor(7, dtype = tf.float32))
print(tf.cast(7, dtype = tf.float32))
print(tf.constant(7.0))
print(tf.constant(7, dtype = tf.float32))

# numpy의 ndarray와 tensor 사이에 type 자동 변환됨
import numpy as np
arr = np.array([1,2])
print(arr, type(arr))
tfarr = tf.add(arr, 5)    # 5가 자동으로 tensor로 변환
print(tfarr)
print(tfarr.numpy())      # 텐서플로우를 벗겨주는거야

print(np.add(tfarr, 3))   # 넘파이 타입으로 자동 형변환돼서 나오는거야
tf.print(tfarr)
print(tfarr)
print(tfarr.numpy())
print(np.add(tfarr,3))

# 텐서플로는 텐서를 Graph 영역 내에서 실행하는 것이 일반적이다.
g1 = tf.Graph()     # 별도의 그래프 생성
with g1.as_default():
  c1 = tf.constant(1, name = 'c_one')
  print(c1)
  print(type(c1))
  print(c1.op.node_def)