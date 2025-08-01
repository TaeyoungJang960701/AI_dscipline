import numpy as np
import random

# array 관련 문제
# 1) step1 : array 관련 문제

#  정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.

# < 출력 결과 예시>

# 1행 합계   : 0.8621332497162859

# 1행 최댓값 : 0.3422690004932227

# 2행 합계   : -1.5039264306910727

# 2행 최댓값 : 0.44626169669315

# 3행 합계   : 2.2852559938172514

# 3행 최댓값 : 1.5507574553572447
row = 5
column = 4
bb = np.random.randn(row, column) # 정규분포(가우시안분포 - 중심극한정리)를 따르는 난수
print(bb)
bb_row_max = []
bb_row_sum = []
for r in range(row):
    row_max_value = np.max(bb[r])
    bb_row_max = np.append(bb_row_max, row_max_value)

for r in range(row):
    row_sum_value = np.sum(bb[r])
    bb_row_sum = np.append(bb_row_sum, row_sum_value)
print('각 행의 최대값 : ',bb_row_max)
print('각 행의 합계 : ', bb_row_sum)

# 2) step2 : indexing 관련문제

#  문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오.
#    조건1> 36개의 셀에 1~36까지 정수 채우기
#    조건2> 2번째 행 전체 원소 출력하기 
#               출력 결과 : [ 7.   8.   9.  10.  11.  12.]
#    조건3> 5번째 열 전체 원소 출력하기
#               출력결과 : [ 5. 11. 17. 23. 29. 35.]
#    조건4> 15~29 까지 아래 처럼 출력하기
#               출력결과 : 
#               [[15.  16.  17.]
#               [21.  22.  23]
#               [27.  28.  29.]]

dim = 6
A = np.arange(1,37).reshape(dim,dim)
# 조건 1
print(A) 
# --------------------------------------------


A_2nd_row = []
A_5th_col = []
A_3rd_district = []

for i in range(dim):
    A_2nd_row = np.append(A_2nd_row, A[1,i])
# 조건 2
print(A_2nd_row) 
# --------------------------------------------


for i in range(dim):
    A_5th_col = np.append(A_5th_col, A[i,4])
# 조건 3
print(A_5th_col) 
# --------------------------------------------


for r in range(3):
    for c in range(3):
        A_3rd_district = np.append(A_3rd_district, A[r+2][c+2])

# 조건 4
print(A_3rd_district.reshape(3,3)) 
# --------------------------------------------


# 문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.

#      조건1> 20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 
#       두 번째 열부터는 1씩 증가시켜 원소 저장하기

#      조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
row = 6
col = 4

rand_1st_row = np.random.choice(range(20,101),row).T
ascending_matrix = []
for i in range(col):
    new_col = rand_1st_row + i
    ascending_matrix.append(new_col)
ascending_matrix = np.array(ascending_matrix).T

# 조건 1
# print(rand_1st_row)
print(ascending_matrix)
# --------------------------------------------


# 조건 2
ascending_matrix[0] = [1000] * col
ascending_matrix[-1] = [6000] * col
print(ascending_matrix)

# --------------------------------------------
# Q1) 브로드캐스팅과 조건 연산
# 다음 두 배열이 있을 때,
# a = np.array([[1], [2], [3]])
# b = np.array([10, 20, 30])
# 두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.
# 그 결과에서 값이 30 이상인 요소만 골라 출력하시오.

a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
c = a*b
print(c)
print(np.where(c >= 30 , c, ''))
# print(c[c>=30]) 이거는 1차원 벡터로 나온다 나중에 쓸모잇겟지
# --------------------------------------------
# Q2) 다차원 배열 슬라이싱 및 재배열
#  - 3×4 크기의 배열을 만들고 (reshape 사용),  
#  - 2번째 행 전체 출력
#  - 1번째 열 전체 출력
#  - 배열을 (4, 3) 형태로 reshape
#  - reshape한 배열을 flatten() 함수를 사용하여 1차원 배열로 만들기

aa = np.arange(1,13).reshape(3,4)
print(aa)
print(aa[1,:])
print(aa[:,0])
aa = aa.reshape(4,3)
print(aa)
print(aa.flatten())

# --------------------------------------------
# Q3) 1부터 100까지의 수로 구성된 배열에서 3의 배수이면서 5의 배수가 아닌 값만 추출하시오.
# 그런 값들을 모두 제곱한 배열을 만들고 출력하시오.
zero_100 = 
# --------------------------------------------
# Q4) 다음과 같은 배열이 있다고 할 때,
# arr = np.array([15, 22, 8, 19, 31, 4])
# 값이 10 이상이면 'High', 그렇지 않으면 'Low'라는 문자열 배열로 변환하시오.
# 값이 20 이상인 요소만 -1로 바꾼 새로운 배열을 만들어 출력하시오. (원본은 유지)
# 힌트: np.where(), np.copy()
# --------------------------------------------
# Q5) 정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개를 만들고, 상위 5% 값만 출력하세요.
# 힌트 :  np.random.normal(), np.percentile()