# tips.csv로 요약 처리 후 시각화
import pandas as pd
import matplotlib.pyplot as plt

tips = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv')
print(tips.info())
tips['gender'] = tips['sex']
print(tips.info())
del tips['sex'] # 칼럼에 섹스를 젠더로 바꾼거야 젠더는 섹스로 섹스랑 똑같은 젠더 칼럼 추가하고 섹스를 지운거지
print(tips.head(3))
print('-' * 60)

# tip ratio를 한번 구해보자 : 파생 변수로 다룰게
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips.head(3))
print('-' * 60)
tip_pct_group = tips['tip_pct'].groupby([tips['gender'], tips['smoker']])
print(tip_pct_group.sum())
print(tip_pct_group.max())
print(tip_pct_group.min())

result = tip_pct_group.describe()
print(result)

print('-' * 60)
print(tip_pct_group.agg('sum'))
print(tip_pct_group.agg('mean'))
print(tip_pct_group.agg('var'))
print('-' * 60)

# 사용자 정의 함수
# 이것도 플롯에 추가해보려 하는거야 그냥 아무거나 만들어본거지
def myFunc(group):
    diff = group.max() - group.min()
    return diff

result2 = tip_pct_group.agg(['var','mean','max','min', myFunc])
print('result2 : ', result2)
result2.plot(kind= 'barh', title = 'agg func result',stacked = True)
plt.show()