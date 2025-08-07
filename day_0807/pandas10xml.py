# XML로 제공되는 날씨자료 처리
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame

# url = 'http://www.kma.go.kr/XML/weather/sfc_web_map.xml'
# # data = urllib.request.urlopen(url).read()
# # print(data.decode('utf8'))
# soup = BeautifulSoup(urllib.request.urlopen(url),'xml')
# print(soup)
# localTag = soup.find_all('local')

# data = []
# for loc in localTag:
#     city = loc.text
#     temp = loc.get('ta')
#     data.append([city,temp])

# df = pd.DataFrame(data, columns = ['지역','온도'])
# print(df)
# df.to_csv('wether.csv', index = False)

df = pd.read_csv('wether.csv')
print(df.head(2))
print(df[0:2])
print('-' * 60)
print(df.tail(2))
print(df[-2:len(df)])
print('-' * 60)
print(df.iloc[0:2, :])
print('-' * 60)
print(df.loc[1:3],['온도'])
print('-' * 60)
print(df.info())
print('-' * 60)
print(df['온도'] >= 30)
print(df.sort_values[df['온도'] >= 32])
print(df.sort_values(['온도'], ascending = True ))