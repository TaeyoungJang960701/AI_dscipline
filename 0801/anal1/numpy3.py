# 배열 연산
import numpy as np

x=np.array([[1,2],[3,4]],dtype=np.float64)
y=np.arange(5,9).reshape(2,2)
y=y.astype(np.float64)
print(x, x.astype, x.dtype)
print(y, y.astype, y.dtype)

# 요소별 합
print(x+y)          # python의 산술 연산자 
print(np.add(x,y))  # numpy의 유니버셜 함수(범용 함수)
# np.subtract, np.multiply, np.divide
import time
big_arr=np.random.rand(1000000)
start=time.time()
sum(big_arr) # python 함수
end=time.time()
print(f"sum():{end-start:.6f}sec")
start=time.time()
np.sum(big_arr) # numpt 함수
end=time.time()
print(f"sum():{end-start:.6f}sec")
# numpt 함수연산 속도>>python 함수연산 속도

# 요소별 곱
print(x)
print(y)
print(x*y)
print(np.multiply(x,y))

print(x.dot(y)) # 내적 연산

# 8/1 4교시 수업
print()
v = np.array([9,10])
w = np.array([11,12])
print(v*w)
print(v.dot(w))
print(np.dot(v,w))
print(np.dot(x,v))

print(np.dot(x, y))

print('유용한 함수 ------------')
print(x)
print(np.sum(x, axis = 0)) #axis가 뭔소리야? 0이니까 column 계산 단위를 설정한거 
print(np.sum(x, axis = 1)) #여기선 1이니까 row로 계산

print(np.min(x),'',np.max(x)) # x 데이터에서 가장 큰 수와 가장 작은 수를 가져와
print(np.argmin(x),'',np.argmax(x)) # 인덱스 반환
print(np.cumsum(x)) # x의 원소들의 누적합
print(np.cumprod(x)) # 누적곱
print()
names = np.array(['tom','james', 'oscar','tom','oscar']) # 일부러 여러개 중복되게 설정
#이거 네임스가 톰 제임스 오스카로 프린트 해라 햇는데 출력창은 제임스 오스카 톰인게 그냥 알파벳 순임
names2 = np.array(['tom','page', 'john']) # 일부러 톰이 중복되게 해놨음
print(np.unique(names))
print(np.intersect1d(names,names2)) # 두 변수의 '교집합'을 가져옴
print(np.intersect1d(names,names2, assume_unique = True)) # 뒤의 어숨 유니크는 중복을 허용한다 라는 문구
print(np.union1d(names,names2)) # 기본적으로 중복을 거부하네
# help(np.unique) 필요하면 주석 떼고 쳐보자 
print('\n전치(Transpose)') # 전치행렬
print(x)
print(x.T)
arr = np.arange(1,16).reshape((3,5))
print(arr)
print(arr.T)
print(np.dot(arr.T,arr)) # 임의 행렬 A와 A의 전치행렬의 곱은 정방행렬임 (예를들면 3x3 행렬)

print(arr.flatten()) # 행렬을 1행으로 늘어뜨려주네
print(arr.ravel()) 