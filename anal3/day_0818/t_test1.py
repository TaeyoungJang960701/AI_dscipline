# 집단 차이분석 : 평균 또는 비율 차이를 분석 :
# 모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있ㄷ.
# T test와 ANOVA의 차이분석두 집단 이하의 변수에 대한 평균차이를 검정할 경우 T test를 사용하여 검정통계량
# 세 집단 이상의 변수에 대한 평균차이를 검정할 경우에는 ANOVA를 이용하여 검정통계

# 핵심 아이디어 :
# 집단 평균차이(분자)와 집단 내 변동성(표준편차, 표준오차 등, 분모)을 비교하여,
# 차이가 데이터의 불확실성(변동성)에 비해 얼마나 큰지를 계산한다
# T 분포는 표본 평균을 이용해 정규분포의 평균을 해석할 때 많이 사용한다
# 대개의 경우 표본의 크기는 30개 이하일 때 T분포를 따른다
# T 검정은 '두개 이하 집단의 평균의 차이가 우연에 의한 것인지 통계적으로 유의한 차이를 판단하는 통계적 절차다.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 실습 1 - 단일 표본 T검정
# 어느 남성 집단의 평균키 검정
# 귀무가설(H0) : 집단의 평균 키가 177이다. (모수)
# 대립가설(H1) : 집단의 평균 키가 177이 아니다. 
one_sample = [167.0, 182.7, 160.6, 176.8, 185.0]
print(np.array(one_sample).mean())  # 174.42
# 177.0과 174.42 는 평균의 차이가 있느냐?
result = stats.ttest_1samp(one_sample, popmean = 179)   # t test 이게 기본 클래스로 들어잇나봐 파이선 라이브러리에
print('statistics : %.3f, pvalue : %.3f'%result)
# statistics : -0.555, pvalue : 0.608
# pvalue가 0.05보다 크다 - > 귀무가설 채택 
# plt.boxplot(one_sample)
# plt.xlabel('data')
# plt.ylabel('value')
# plt.show()
# plt.close()

# 실습 2 - 단일 모집단의 평균에 대한 가설검정 (one samples t test)
# 중학교 1학년 1반의 학생들의 시험결과가 담긴 파일을 읽어 처리 국어 점수 평균검정 student
# 귀무가설(H0) : 학생들의 국어 점수의 평균은 80이다.
# 대립가설(H1) : 학생들의 국어 점수의 평균은 80이 아니다.

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
print(data.head(3))

print(data.describe())
print('-' * 100)

# 정규성 검정 : one-sample t-test는 필수적이지 않음
#  
print('정규성 검정 : ',stats.shapiro(data.국어))
# 정규성 검정 :  ShapiroResult(statistic=np.float64(0.8724170171507788), pvalue=np.float64(0.01295975332132026))
res = stats.ttest_1samp(data.국어, popmean = 80) # pvalue=(0.01295975332132026)
print('-' * 100)
# pvalue가 0.05보다 작으므로 정규성을 만족하지 못한다
# 정규성 위배는 데이터 재가공이 필요, Willcoxon Signed-Rank test를 써야 더 안전(신뢰도가 높다?)
# Willcoxon Signed-Rank test는 정규성을 가정하지 않는다
from scipy.stats import wilcoxon

wilcox_res = wilcoxon(data.국어 - 80)   # 평균인 80점과 비교하는거야
print('wilcox_res : ',wilcox_res)
# wilcox_res :  WilcoxonResult(statistic=np.float64(74.0), pvalue=np.float64(0.39777620658898905))
print('-' * 100)

print('statistics : %.3f, pvalue : %.3f'%res)
# pvalue값이 0.199이기에 0.05보다 크다 -> 귀무가설 채택

# 해석 : 정규성은 부족하지만 t-test와 Wilcoxon은 같은 결과를 얻었다. 표본수가 커지면 결과는 달라질 수 있다.
# 정규성 위배가 있었음에도 t-test 결과는 신뢰할 수 있다.
