import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np

# 재색인
# Series의 재색인
data = Series([1, 3, 2], index=(1, 4, 2))
print(data)
data2 = data.reindex((1, 2, 4)) #1, 2, 4 순으로 재 인덱싱
print(data2)

# 재색인 할 때 값 채우기
data3 = data2.reindex([0, 1, 2, 3, 4, 5])
print(data3)    #없는 값들은 모두 NaN으로 뜸
# 대응값이 없는(NaN) 인덱스는 결측값인데 777로 채우기
data3 = data2.reindex([0, 1, 2, 3, 4, 5], fill_value=777)
print(data3)
# 결측값이 없을경우 이전 값으로 다음값을 채움
print()
data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='ffill')   #method='pad' 로도 가능!
print(data3)
# 결측값이 없을경우 앞의 값으로 다음값을 채움
data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='bfill')   #method='backfill' 로도 가능!
print(data3)

# bool 처리, 슬라이싱 관련 method : loc(), iloc()
df = DataFrame(np.arange(12).reshape(4, 3), index=['1월', '2월', '3월', '4월'],
               columns=['강남', '강북', '서초'])            #4행3열짜리 데이터 생성
print(df)
print(df['강남'])       #강남 열만 출력
print(df['강남'] > 3)   #강남 열 중 3을 초과하는 것만
print(df[df['강남'] > 3])   #강남 중 데이터 값이 3을 초과하는 것만
print()
print(df.loc[:'2월'])
print(df.loc[:'2월',['서초']])
print()
print(df.iloc[2])
print(df.iloc[2,:])
print(df.iloc[:3])
print(df.iloc[:3,2], type(df.iloc[:3,2]))
print(df.iloc[:3,1:3])

# <기초 알고리즘>
# 알고리즘은 문제를 해결하기 위한 일련의 단계적 절차 또는 방법
# 어떤 문제를 해결하기 위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어를 순서대로 나열한 것이라 할 수 있음.

# --파일 test1.py--
#<알고리즘>
#문제를 해결하기 위한 일련의 단계적 절차 또는 방법
#어떤 문제를 해결하기 위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어를 순서대로 나열한 것이라 할 수 있음.
#컴퓨터 프로그램을 만들기 위한 알고리즘은 계산과정을 최대한 구체적이고 명료하게 작성해야 한다.
#'문제 -> 데이터입력 -> 알고리즘으로 처리 -> 결과 출력'이 알고리즘의 기본



#문1) 1 ~ 10(n) 까지의 연속된 정수의 합 구하기
def totFunc(n):     #방법 1 -> O표기법으로는 시간복잡도 O(n)
    tot = 0
    for i in range(1, n + 1):
        tot = tot + i
    return tot

print(totFunc(100))

def totFunc2(n):    #방법 2 -> O표기법으로는 시간복잡도 O(1), 방법 1보다 더 빠름
    return n * (n + 1) // 2     #덧셈 후 곱셈 후 나눗셈

print(totFunc2(100))

#주어진 문제를 푸는 방법은 다양하다. 어떤 방법이 더 효과적인지 알아내는 것이 '알고리즘 분석'
#'알고리즘 분석' 평가 방법으로 계산 복잡도 표현 방식이 있음.
# 1) 공간 복잡도 : 메모리 사용량 분석
# 2) 시간 복잡도 : 처리 시간을 분석
# O(빅 오) 표기법 : 알고리즘의 효율성을 표현해주는 표기법


#문2) 임의 정수들 중 최대값 찾기
#입력 : 숫자 n개를 가진 list
#출력 : 숫자 n개 중 최대값을 출력
def findMaxFunc(a):                 #방법1) 시간복잡도 O(n)
    maxValue = a[0]
    for i in range(1, len(a)):
        if (a[i] > maxValue):
            maxValue = a[i]
    return maxValue

d = [17, 92, 11, 33, 55, 7, 27, 42]
print(findMaxFunc(d))

#최대값 위치(인덱스) 반환
def findMaxFunc2(a):                 #방법2) 시간복잡도 O(n)
    maxValue = 0
    for i in range(1, len(a)):
        if (a[i] > a[maxValue]):
            maxValue = i
    return maxValue

d = [17, 92, 11, 33, 55, 7, 27, 42]
print(findMaxFunc2(d))
