'''
기술 통계의 목적은 데이터를 수지, 요약, 정리, 시각화
- 도수분포표 frequency distribution : 데이터를 구간별로 나눠 빈도를 정리한 표
    - 이를 통해 데이터의 분포를 한 눈에 파악할 수 있다 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# 기술 통계를 한번 살짝 보자

# Step 1 : 데이터를 읽어서 DataFrame에 저장
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv')
print(df.head())

# Step 2 : max, min
min_height = df['키'].min()
max_height = df['키'].max()
print(f'최대값 : {max_height}, 최소값 : {min_height}')

# Step 3 : 구간 설정 - 계급 나눌 수도 잇다 (cut)
bins = np.arange(156, 195, 5)   # 156부터 195까지 5구간으로 나누겟다
print(bins)
df['계급'] = pd.cut(df['키'], bins = bins, right = True, include_lowest= True)
# 각 데이터가 어떤 구간에 들어가 있는지까지도 이렇게 프린트 해서 볼 수 있다
print(df)
print(df.head(3))
print(df.tail(3))

# Step 4 : 각 계급의 중앙값을 한번 보자     (156 + 156) / 2 -> 중앙값 예시
df['계급값'] = df['계급'].apply(lambda x:int(x.left + x.right) / 2)
print(df.head(5))

# Step 5 : 도수를 계산해보자
freq = df['계급'].value_counts().sort_index()
print(df.head())

# Step 6 : 상대 도수를 계산해보자(전체 데이터에 대한 비율)
relative_freq = (freq / freq.sum()).round(2)
print(relative_freq)

# Step 7 : 누적 도수 계산 - 계급별 도수를 누적
cum_freq = freq.cumsum()
print(cum_freq.head())

# Step 8 : 위의 자료를 종합해 도수 분포표를 작성해보자
dist_table = pd.DataFrame({
    # '156 ~ 161' 이런 모양을 출력해 볼 수 있다
    '계급' : [f'{int(interval.left)} ~ {int(interval.right)}' for interval in freq.index],
    # 계긊의 중간값
    '계급값' : [int((interval.left + interval.right) / 2) for interval in freq.index],
    '도수' : freq.values,
    '상대도수' : relative_freq.values,
    '누적도수' : cum_freq.values,
})
print('도수 분포표\n',dist_table.head())

# 요게 기술통계로 써보는법이다

# Step 9 : 히스토그램 그려보기
plt.figure(figsize = (8,5)) # 가로 8인치 세로 5인치
plt.bar(dist_table['계급값'], dist_table['도수'], width = 5, 
        color = 'cornflowerblue', edgecolor = 'black')
plt.title('학생 50명 히스토그램', fontsize = 16)
plt.xlabel('키(계급값)')
plt.ylabel('도수')
plt.xticks(dist_table['계급값'])
plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
plt.show()