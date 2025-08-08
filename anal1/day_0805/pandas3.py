import numpy as np
from pandas import Series, DataFrame
import pandas as pd

s1 = Series([1,2,3], index = ['a','b','c'])
s2 = Series([4,5,6], index = ['a','b','d'])
print(s1)
print(s2)

print(s1 + s2)
print(s1.add(s2))
print(s1.multiply(s2)) # sub, div 다 가능해
print()
df1 = DataFrame(np.arange(9).reshape(3,3), columns = list('kbs'), index = ['서울','대전','대구'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns = list('kbs'), index = ['서울','대전','제주','수원'])
print(df1)
print(df2)
print(df1 + df2) # 각자 갖고 잇는 겹치는 요소들끼리만 연산함 나머지 중복되지 않는것들은 NaN
print(df1.add(df2, fill_value = 0))   # NaN 은 0으로 채운 후 연산에 참여
print()
ser1 = df1.iloc[0]
print(ser1)
print(df1 - ser1)  # Broadcasting 연산
print('--- 결측치, 기술적 통계 관련 함수 ---')
# 결측치
df = DataFrame([[1.4,np.nan],[7,-4.5],[np.nan, None], [0.5,-1]], columns = ['one','two'])
print(df)
print(df.isnull()) # 널값은 참으로 나와
print(df.notnull()) # 여기선 거짓으로 나오지
print(df.drop(0)) # NaN과 관계 없는거야 0은 지워버리는거야 특정 값만 지우는거지
print(df.dropna()) # NaN 값이 포함된 모든 행 삭제 
print(df.dropna(how = 'any')) # 이게 drop na의 디폴트 값이래 NaN 잇으면 다 조져버려
print(df.dropna(how = 'all')) # 얘는 요소값이 모두 NaN이어야 조지는거래
print(df.dropna(subset = ['one'])) # 이건 어떤 일이 일어나는거지
print(df.dropna(axis = 'rows'))
print(df.dropna(axis = 'columns'))
print(df.fillna(0))

print('기술 통계 관련 메서드')
print(df.sum())  # 열의 합 - NaN은 연산에 참여하지 않음
print(df.sum(axis = 0))
print(df.sum(axis = 1))  # 썸 디바이드 타임즈 다 잇다
print(df.describe())   # 요약 통계량을 보여준다 뭐 데이터 개수 평균 산평 분산 다잇대
print(df.info())   # 구조 확인

print('재구조화, 구간 설정, 그룹별 함수 agg 함수')
df = DataFrame(1000 + np.arange(6).reshape(2,3), 
               index = ['서울','대전'], columns = ['2020','2021','2022'])
print(df)
print(df.T)
# stack, unstack
df_row = df.stack()   # column -> row로 변경. 열 쌓기
print(df_row)

df_col = df_row.unstack()   # 행 -> 열로 복원한다
print(df_col)

# 구간 설정 : 연속형 자료를 범주화
price = [10.3, 5.5, 7.8, 3.6]   # 
cut = [3, 7, 9, 11]   # 구간 기준값
result_cut = pd.cut(price, cut)   # 판다스.컷(데이터, 구간값)
print(result_cut)
print(pd.value_counts(result_cut))

datas = pd.Series(np.arange(1,1001))
print(datas.head(6)) # 앞의 6개 구간만 볼게
print(datas.tail(6)) # 뒤의 6개 구간만 볼게
result_cut2 = pd.qcut(datas,3)
print(result_cut2)
print(pd.value_counts(result_cut2))

print('-----------------')
group_col = datas.groupby(result_cut2)
# print(group1_col)
print(group_col.agg(['count','mean','std','min']))

def myFunc(gr):   # agg 함수 쓰면 되긴 하지만 내가 한번 해보는거야 그냥
    return{
        'count':gr.count(),
        'mean':gr.mean(),
        'std':gr.std(),
        'min':gr.min(),
        }
print(group_col.apply(myFunc))
print(group_col.apply(myFunc).unstack())