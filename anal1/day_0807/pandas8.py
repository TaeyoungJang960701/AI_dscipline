import pandas as pd
from pandas import DataFrame
from bs4 import BeautifulSoup
import urllib       # 연습용, 코드가 장황함
import requests     # 실전용, 코드가 간결함

# 웹 스크래핑
url = 'https://www.kyochon.com/menu/chicken.asp'
response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text,'lxml')
print(soup)

# 메뉴 이름 추출
names = [tag.text.strip() for tag in soup.select('dl.txt > dt')]
print(names)

#가격 이름 추출
prices = [int(tag.text.strip().replace(',',''))for tag in soup.select('p.money')]
print(prices)

df = pd.DataFrame({'상품명':names, '가격':prices})
print(df.head(3))
print('가격 평균:', round(df['가격'].mean(),2))
print(f"가격 평균: {df['가격'].mean():.2f}")
print('가격 표준편차:', round(df['가격'].std(),2))

