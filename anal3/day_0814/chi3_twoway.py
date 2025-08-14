# 이원 카이 제곱검정 - 교차분할표를 사용
# 변인이 두 개 - 독립성 또는 동질성 검사
# 독립성(관련성)검정- 동일 집단의 두 변인(학력 수준과 대학 진학 여부)을 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.

# 실습 : 교육수준과 흡연율 간의 관련성 분석 smoke.csv
# 귀무 : 교육수준과 흡연율 간의 관련이 없다. (독립이다, 연관성이 없다)
# 대립 : 교육수준과 흡연율 간의 관련이 있다. (독립이 아니다, 연관성이 있다)

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv')
print(data.head())
print('-' * 100)
print(data['education'].unique())
print(data['smoking'].unique())

# 학력별 흡연 인원수를 위한 교차표 작성
ctab = pd.crosstab(index = data['education'], columns = data['smoking'])
# ctab = pd.crosstab(index = data['education'], columns = data['smoking'], normalizing = True)

print(ctab)
ctab.index = ['대학원졸','대졸','고졸']
ctab.columns = ['과흡연','보통','노담']
print(ctab)
print('-' * 100)

chi2, p, dof, _ = stats.chi2_contingency(ctab)
msg = 'test statics : {}, p-value : {}, df : {}'
print(msg.format(chi2, p, dof)) 
# test statics : 18.910915739853955, p-value : 0.0008182572832162924, df : 4
# 결론 : p-value(0.00082) < 유의수준(0.05) 이므로 귀무가설 기각
# 따라서 '교육 수준과 흡연율 간의 관련이 없다' 가설은 기각하고 
# '교육 수준과 흡연율 간의 관련이 있다' 가설을 채택함
print('-' * 100)

print('음료 종류와 성별 간의 선호도 차이 검정 --------')
# 남성과 여성의 음료 선호는 서로 관련이 있을까?
# 귀무(H0) : 성별과 음료에 대한 선호는 서로 관련이 없다.(성별에 따라 선호가 같음)
# 대립(H1) : 성별과 음료에 대한 선호는 서로 관련이 있다.(성별에 따라 선호가 다름)

data = pd.DataFrame({
    '게토레이' : [30,20],
    '포카리' : [20,30],
    '비타오백' : [10,30]
}, index = ['남성','여성'])
print(data)

chi2, p, dof, expected = stats.chi2_contingency(data)
print('카이제곱 통계량 : ', chi2)
print('유의확률(p값) : ', p)
print('자유도(DoF) : ', dof)
print('기대도수(expected) : ', expected)

# 카이제곱 통계량 :  11.375
# 유의확률(p값) :  0.003388052521834713
# 자유도(DoF) :  2
# 기대값(expected) :  [21.42857143 21.42857143 17.14285714]
#                    [28.57142857 28.57142857 22.85714286]
# 결론 : p값이 0.05보다 작으므로 귀무 가설은 기각이다. 
# 따라서 성별에 따른 음료 선호도는 종속적이다.

# 시각화 : heatmap
# 히트맵(heatmap)은 데이터 분석에서 많이 사용되는 시각화 기법 중 하나입니다. 
# 일반적으로 데이터셋의 값을 색상으로 나타내어 시각적으로 이해하기 쉽게 만들어줍니다.

import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family = 'Malgun Gothic')
sns.heatmap(data, annot = True, fmt = 'd', cmap = 'Blues')
plt.title('성별에 따른 음료 선호 (빈도)')
plt.xlabel('음료 종류')
plt.ylabel('성별')
plt.show()

