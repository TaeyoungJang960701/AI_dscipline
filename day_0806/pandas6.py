# pandas로 파일 저장

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

items = {'apple' : {'count':10, 'price' : 1500}, 'orange':{'count':4, 'price':700}}
df = pd.DataFrame(items)
print(df)

# df.to_clipboard()
# print(df.to_html())
print(df.to_json())

df.to_csv('result1.csv', sep = ',', index = False, header = False)

data= df.T
print(data)
df.to_csv('result.csv', sep = ',', index = True, header = True)

# 엑셀 관련
df2 = pd.DataFrame({'name':['Alice','Bob','Oscar'],'age':[24,36,33],'city':['seoul','suwon','busan']})
print(df2)

# 엑셀 저장
df2.to_excel('result2.xlsx', index = False, sheet_name = 'mysheet')
# 엑셀 읽기
exdf = pd.ExcelFile('result2.xlsx')
print(exdf.sheet_names)
dfexcel = exdf.parse('mysheet')
print(dfexcel)