# 이원카이제곱
# 동질성:
# 검정 두 집단의 분포가 동일한가 다른 분포인가를 검증하는 방법이다 두 집단이상에서 각 
# 동일 한가를 검정하게 된다 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출

# 검정실습1: 교육방법에 따른 교육생들의 만족도 분석 동질성 검증 survey_method csv

import pandas as pd
import scipy.stats as stats
# Scientific Python의 줄임말
# 과학 계산을 위한 라이브러리
# numpy 기반으로 수치연산, 선형대수, 최적화, 신호처리, 통계 등 다양한 기능 제공
# SciPy 라이브러리의 stats 모듈을 불러오는 코드

data=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
# print(data.head(3))
# print(data['method'].unique())  # [1 2 3]
# print(set(data['survey']))  # {1, 2, 3, 4, 5}

ctab=pd.crosstab(index=data['method'],columns=data['survey'])
ctab.columns=['매우만족','만족','보통','불만족','매우불만족']
ctab.index=['방법1','방법2','방법3']
print(ctab)

chi2,p,ddof,_=stats.chi2_contingency(ctab)
msg='test statistic:{},p-value:{},df:{}'
print(msg.format(chi2,p,ddof))
# test statistic:6.544667820529891,p-value:0.5864574374550608,df:8
# 해석: 유의수준 0.05 < p-value: 0.5864574374550608 이므로 귀무가설 채택

print('=='*20)

# 질성검정실습2) 연령대별sns 이용률의 동질성검정
# 20대에서40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용현황을 조사한자료를 바탕으로 연령대별로 홍보
# 전략을 세우고자한다.
# 연령대 별로 이용현황이 서로 동일한지 검정 해 보도록 하자.

# 귀무가설(H0) : 사용자의 연령대와 그에 따른 SNS 이용 현황은 관련이 없다 (독립적이다)
# 대립가설(H1) : 사용자의 연령대와 그에 따른 SNS 이용 현황은 관련이 있다 (종속적이다)

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')

print(data2.head())

print(data2['service'].unique())  # ['F' 'T' 'K' 'C' 'E']
print(data2['age'].unique())      # [1 2 3]

ctab2 = pd.crosstab(index = data2['age'], columns = data2['service'], margins = True)
print(ctab2)
print(data2.head())

chi2,p,ddof,_=stats.chi2_contingency(ctab2)
msg='test statistic:{},p-value:{},df:{}'
print(msg.format(chi2,p,ddof))  # test statistic:102.75202494484225,p-value:3.916549763057839e-15,df:15
# 결론 : 유의수준 0.05 이하의 p-value, 3.916549763057839e-15 이므로 귀무가설을 기각한다

# 사실 위 데이터의 경우는 샘플 데이터이다. 하지만 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본을 추출해보자.

sample_data = data2.sample(n = 50, replace = True, random_state = 1)
print(len(sample_data))
print('-' * 100)
ctab3 = pd.crosstab(index = sample_data['age'], columns = sample_data['service'], margins = True)
print(ctab3)
print('-' * 100)
print(data2.head())
chi2,p,ddof,_=stats.chi2_contingency(ctab3)
msg='test statistic:{},p-value:{},df:{}'
print(msg.format(chi2,p,ddof))
print('-' * 100)