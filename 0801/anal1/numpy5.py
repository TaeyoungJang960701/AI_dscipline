# 배열에 행 또는 열 추가
import numpy as np
import random

aa = np.eye(3)
print('aa : \n',aa)
bb = np.c_[aa, aa[2]]
print(bb)
cc = np.r_[aa, [aa[2]]]
print(cc)

# reshape
a = np.array([1,2,3])
print('np.c_ : \n',np.c_[a])
a.reshape(3,1)
print(a)

print('---- append, insert, delete ----')
# 1차원
print(a)
b = np.append(a,[4,5])
print(b)
c = np.insert(a, 2, [6,7])
print(c)
# d = np.delete(a, 1)
# d = np.delete(a, [1])
d = np.delete(a, [1,2])
print(d)

print()
# 2차원
aa = np.arange(1,10).reshape(3,3)
print(aa)
print(np.insert(aa, 1, 99))   # 삽입 후 차원 축소가 이루어진다
print(np.insert(aa, 1, 99, axis = 0))
print(np.insert(aa, 1, 99, axis = 1))

print(aa) # 앞에서 지지고 볶고 해도 원본 aa는 살아잇다
bb = np.arange(10,19).reshape(3,3)
print(bb)
cc = np.append(aa, bb) # 추가 후 차원 축소
print(cc)
cc = np.append(aa, bb, axis = 0)
print(cc)
cc = np.append(aa, bb, axis = 1)
print(cc)

print('np.append 연습')
print(np.append(aa, 88))
print(np.append(aa, [[88,88,88]], axis = 0))
print(np.append(aa, [[88],[88],[88]], axis = 1))

print()
print(np.delete(aa,1)) # 삭제 후 축소
print(np.delete(aa,1,axis = 0))
print(np.delete(aa,1,axis = 1))

# 조건 연산 where(조건, 참, 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
condData = np.array([True,False,True])
result = np.where(condData, x, y)
print(result)
print()
aa = np.where(x >= 2)
print(aa) # (array([1,2]),)    index로 나오네?
print(x[aa]) 
print(np.where(x >= 2, 'T','F'))
print(np.where(x >= 2, x, x + 100))

bb = np.random.randn(4,4) # 정규분포(가우시안분포 - 중심극한정리)를 따르는 난수
print(bb)
print(np.where(bb > 0, 7, bb)) # 이게 where 함수에 대한 직관적인 설명이네

print('배열 결함/분할')
kbs = np.concatenate([x, y]) # 배열 결합
print(kbs)
print()
x1, x2 = np.split(kbs, 2)
print(x1)
print(x2)
print()
a = np.arange(1,17).reshape(4,4)
print(a)
x1, x2 = np.hsplit(kbs, 2)
print(x1)
print(x2)
print()
x1, x2 = np.hsplit(kbs, 2)
print(x1)
print(x2)
print('복원, 비복원 추출')
datas = np.array([1,2,3,4,5,6,7])

# 복원 추출
for _ in range(5): # 이건 그냥 넘파이 내의 함수 말고 조건문 만들어서 쓴거
    print(datas[np.random.randint(0, len(datas) - 1)], end = '')

# 비복원 추출 전용 함수 - sample()
print()
print(random.sample(datas.tolist(), 5)) # 랜덤은 넘파이 함수가 아니니까 위에 따로 임포트 해줫어

print('-----------------')
# 추출 함수 : choice()
# 복원 추출
print(np.random.choice(range(1,10), 6)) # 복원 추출
# 비복원 추출
print(np.random.choice(range(1,10), 6, replace = False))

# 가중치를 부여한 랜덤 추출
ar = 'air book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar,3,p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])) # 랜덤 추출되는 확률을 항목마다 각각 부여
