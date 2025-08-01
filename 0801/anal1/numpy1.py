# 기본통계함수를 직접 작성하기 : 평균, 분산, 표준편차
grades=[1,3,-2,4]

def grades_sum(grades):
    tot=0
    for g in grades:
        tot += g
    return tot

print(grades_sum(grades))


def grades_ave(grades):
    ave=grades_sum(grades)/len(grades)
    return ave

print(grades_ave(grades))

def grades_variance(grades):
    ave=grades_ave(grades)
    vari=0
    for su in grades:
        vari+=(su - ave)**2
    return vari/len(grades)

print(grades_variance(grades))


def grades_std(grades):
    return grades_variance(grades)**0.5

print(grades_std(grades))


print('**'*10)
import numpy as np # numpy는 계산을 위한 라이브러리
print('합은', np.sum(grades))
print('평균은', np.mean(grades))
# print('평균은', np.average(grades)) 가중평균 주기 가능
print('분산은', np.var(grades))
print('표준편차는', np.std(grades))
