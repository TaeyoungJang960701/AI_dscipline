# JSNO: XML에 비해 가벼우며, 배열에 대한 지식만 있으면 처리 가능

import json
dict = {'name': 'Tom','age': 33, 'score' : ['90','80','100']}
print(type(dict))

print('Json S인코딩 (dict를 Json 모양의 문자열로 변경하는것)')
str_val = json.dumps(dict)
print('str_val:%s'%str_val)
print(type(str_val))
# print(str_val['name']) 
print('json deciding(str을 dict로 변경하는것)-----------')
json_val = json.loads(str_val)
print(type(json))
print(json_val['name'])

for k in json_val.keys():
    print(k)

print('웹문서에서 json 문서 읽기')
import urllib.request as req

url = 'https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json'
plainText = req.urlopen(url).read().decode()
print(plainText)
print(type(plainText))
jsonData = json.loads(plainText)
print(type(jsonData))
print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME']) # LH 강남 3단지 작은도서관
print('-'*60)
# dict의 자료를 읽어라 도서관명, 전화, 주소
# 디비에 가져올것같애
libData = jsonData.get('SeoulLibraryTime').get('row')
# print(libData)
print(libData[0].get('LBRRY_NAME'))
print('-'*60)

datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    # print(name + '\t' + tel + '\t' + addr)
    datas.append([name,tel,addr])

import pandas as pd
df = pd.DataFrame(datas, columns = ['도서관명','전호','주소'])
print(df)
print('-'*60)
