import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df1 = pd.DataFrame({'data1': range(7),'key': ['b', 'b', 'b', 'c', 'a', 'a', 'b']})
print(df1)

df2 = pd.DataFrame({
    'key': ['a', 'b', 'd'],
    'data2': range(3)
})
print("\ndf2:")
print(df2)

# inner join (기본) 머지 해서 합치는걸 얘기햇엇어 그냥 -지피티 이후- 그거보다는 이너는 교집합 아우터는 합집합
print("\nInner Join (기본 merge):")
print(pd.merge(df1, df2, on='key'))

# outer join
print("\nOuter Join:")
print(pd.merge(df1, df2, on='key', how='outer'))

# left join
print("\nLeft Join:")
print(pd.merge(df1, df2, on='key', how='left'))

# right join
print("\nRight Join:")
print(pd.merge(df1, df2, on='key', how='right'))

print('==========공통 칼럼이 없는 경우???==========')
df3=pd.DataFrame({'key2':['a','b','c'],'data2':range(3)})
print(df1)
print(df3)
print(pd.merge(df1,df3,left_on='key',right_on='key2'))

print('=='*20)
print(pd.concat([df1,df3],axis=0))
print()
print(pd.concat([df1,df3],axis=1))
print()

s1 = pd.Series([0,1],index = ['a','b'])
s2 = pd.Series([2,3,4],index = ['c','d','e'])
s3 = pd.Series([5,6], index = ['f','g'])
print(pd.concat([s1,s2,s3],axis = 0))

print('그룹화 : pivot_table')
data = {
    'city':['강남','강북','강남','강북'],
    'year' : [2000,2001,2002,2003],
    'pop' : [3.3,2.5,3.0,2.0]
}
df = pd.DataFrame(data)
print(df)
print(df.pivot(index = 'city', columns = 'year', values = 'pop')) # 연산 대상이 value값인가봐
print()
print(df.set_index(['city','year']).unstack())
print(df.describe())
print('pivot_table : pivot과 groupby의 중간적 성격')
print(df.pivot_table(index = ['city'], values = 'pop'))
print(df.pivot_table(index = ['city'], aggfunc = 'mean'))
print(df.pivot_table(index = ['city','year'], aggfunc = [len,sum]))
print(df.pivot_table(index = 'city', values = 'pop', aggfunc = [len,sum]))
print(df.pivot_table(values = ['pop'], index = ['year'],
                     columns = ['city'], margins = True))
print(df.pivot_table(values = ['pop'], index = ['year'],
                     columns = ['city'], margins = True,fill_value = 0))
print()
hap = df.groupby(['city'])
print(hap)
print(hap.sum(['city']).sum())
print(df.groupby(['city','year']).mean())

# 이거 좀 다시 뜯어보자 졸앗다