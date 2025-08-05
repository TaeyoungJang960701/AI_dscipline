# Broadcasting : 크기가 다른 배열 간의 연산 시 배열의 구조 자동 변환기 된다
# 작은 배열과 큰 배열 연산 시 작은 배열은 큰 배열의 구조를 따른다

import numpy as np

x = np.arange(1,10).reshape(3,3) 
y = np.array([1,0,1]) 
print(x) # x는 3행 3열짜리 매트릭스
print(y) # y는 1행 3열짜리 매트릭스

# 두 배열의 요소 더하기
# 1) 새로운 배열을 이용
z = np.empty_like(x)
print(z)
for i in range(3):
    z[i] = x[i] + y

print(z)

# 2) tile을 이용
kbs = np.tile(y,(3,1))
print('kbs : ',kbs)
z = x + kbs
print(z)

# broadcasting을 이용
# 1D + 1D(같은 길이), 1D + 1D(한쪽 길이 1), (2D + 2D 같은)
kbs = x + y
print(kbs)

a = np.array([0,1,2])
b = np.array([5,5,5])
print(a + b)
print(a + 5)

print('\n넘파이로 파일 입출력을 할 수도 있다(i/o)')
np.savetxt('numpy4etc', x) # binary 형식으로 저장이 된다
np.savetxt('numpy4etc.txt',x)
imsi = np.load('numpy4etc.npy')
print(imsi)

mydatas = np.loadtxt('numpy4etc2.txt',delimiter = ',')
print(mydatas)