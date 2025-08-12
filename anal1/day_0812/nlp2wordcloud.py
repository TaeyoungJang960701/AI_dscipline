# pip install pygame
# pip install pytagcloud
# pip install simplejson
# 동아일보 검색 기능으로 문자열을 읽어 형태소를 분석하고 워드클라우드 출력

from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

# keyword = input('검색어 : ')
keyword = '무더위'
print(keyword)

# 밑에 %포함돼서 드럽게 나온건 인코딩 된거야
target_url = 'https://www.donga.com/news/search?query=%EC%A4%91%EB%B3%B5' + quote(keyword)
print(quote(target_url))
# print(target_url)
# https://www.donga.com/news/search?query=%EC%A4%91%EB%B3%B5무더위
# 이 유알엘로 들어가려 하면 드갈 수 없다 무더위가 인코딩이 안됏어
# 하지만 quote를 쓰면 인코딩이 된 유알엘이 나와
# https%3A//www.donga.com/news/search%3Fquery%3D%25EC%25A4%2591%25EB%25B3%25B5%25EB%25AC%25B4%25EB%258D%2594%25EC%259C%2584
# 이렇게 나와
print(target_url)

source_code = urllib.request.urlopen(target_url)
soup = BeautifulSoup(source_code, 'lxml', from_encoding = 'utf-8')
print(soup)