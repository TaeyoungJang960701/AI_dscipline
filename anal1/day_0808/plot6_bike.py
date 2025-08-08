# 자전거 공유 시스템(워싱턴 DC) 관련 파일로 시각화

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

plt.rc('font', family= 'malgun gothic')     # 한글 깨짐 방지 코드 두줄
plt.rcParams['axes.unicode_minus']= False   # 한글 깨짐 방지 코드 두줄

plt.style.use('ggplot')

# 데이터 수집 후 가공(EDA) - train.csv 영권샘 깃허브에서 가져온 그거
train = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', 
                    parse_dates = ['datetime'])     # 기존의 datetime은 object로 출력됏엇어 이걸 데이트 타임으로 바꿔줫어

print(train.shape)      # 몇 바이 몇인지 알려줘 (10886,12)
print('-' * 60)
print(train.columns)    # 어떤 칼럼을 가지고 있는지 보여줘
print('-' * 60)
print(train.info())     # 칼럼의 형식들을 보여줘
print('-' * 60)
print(train.head(3))
print('-' * 60)
pd.set_option('display.max_columns', 500)
print(train.temp.describe())    # 온도 기준으로 정규분포표 분석해줘
print(train.isnull().sum())     # 널이 포함된 열 확인용 - 널값 갖고잇는애들 가져와봐

# null이 포함된 열 확인용 시각화 모듈
# - 모듈 설치 pip install missingno : 결측치 시각화 모듈

msno.matrix(train, figsize=(12,5))
plt.show()
msno.bar(train,figsize=(12,5)) # 막대 그래프
plt.show()

# 연월일시 데이터로 자전거 대여량 시각화
train['year'] = train['datetime'].dt.year # 연월일시분초 칼럼 생성
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
# print(train.columns) 
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count',
#        'year', 'month', 'day', 'hour', 'minute', 'second'],
#       dtype='object')

# figure,(ax1, ax2, ax3, ax4) = plt.subplots(nrows = 1, ncols =4)
# figure.set_size_inches(15,5)
# sns.barplot(data = train, x = 'year', y = 'count', ax = ax1)
# sns.barplot(data = train, x = 'month', y = 'count', ax = ax2)
# sns.barplot(data = train, x = 'day', y = 'count', ax = ax3)
# sns.barplot(data = train, x = 'hour', y = 'count', ax = ax4)
# ax1.set(ylabel = '건수', title = '연도별 대여량')
# ax1.set(ylabel = '월', title = '월별 대여량')
# ax1.set(ylabel = '일', title = '일별 대여량')
# ax1.set(ylabel = '시', title = '시간별 대여량')
# plt.show()

# Boxplot으로 시각화 - 대여량 - 계절별, 시간별 근무일 여부에 따른 대여량
# 이건 내가한거야

# figure,(ax1,ax2,ax3) = plt.subplots(nrows = 1, ncols = 3)
# figure.set_size_inches(15,5)

# sns.boxplot(data = train, x = 'season', y = 'count', ax = ax1)
# sns.boxplot(data = train, x = 'hour', y = 'count', ax = ax2)
# sns.boxplot(data = train, x = 'workingday', y = 'count', ax = ax3)

# ax1.set(ylabel = '건수', title = '계절별 대여량')
# ax2.set(ylabel = '건수', title = '시간별 대여량')
# ax3.set(ylabel = '건수', title = '근무일 대여량')
# plt.show()

fig, axes = plt.subplots(nrows =2, ncols =2)
fig.set_size_inches(12,10)
sns.boxplot(data = train, y = 'count', orient = 'v', ax = axes[0][0])
sns.boxplot(data =train, y = 'count', x ='season', orient = 'v', ax =axes[0][1]0)
sns.boxplot(data =train, y = 'count', x ='hour', orient = 'v', ax =axes[1][0]0)
sns.boxplot(data =train, y = 'count', x ='workingday', orient = 'v', ax =axes[1][1]0)