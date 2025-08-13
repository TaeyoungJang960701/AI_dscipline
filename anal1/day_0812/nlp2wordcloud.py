# pip install pygame
# pip install pytagcloud
# pip install simplejson
# 동아일보 검색 기능으로 문자열을 읽어 형태소를 분석하고 워드클라우드 출력

from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
from konlpy.tag import Okt
from collections import Counter

# keyword = input('검색어 : ')
keyword = '무더위'
print(keyword)

# 밑에 %포함돼서 드럽게 나온건 인코딩 된거야
target_url = 'https://www.donga.com/news/search?query=%EC%A4%91%EB%B3%B5' + quote(keyword)
print(target_url)
# print(target_url)
# https://www.donga.com/news/search?query=%EC%A4%91%EB%B3%B5무더위
# 이 유알엘로 들어가려 하면 드갈 수 없다 무더위가 인코딩이 안됏어
# 하지만 quote를 쓰면 인코딩이 된 유알엘이 나와
# https%3A//www.donga.com/news/search%3Fquery%3D%25EC%25A4%2591%25EB%25B3%25B5%25EB%25AC%25B4%25EB%258D%2594%25EC%259C%2584
# 이렇게 나와
print(quote(target_url))

source_code = urllib.request.urlopen(target_url)
soup = BeautifulSoup(source_code, 'lxml', from_encoding = 'utf-8')
# print(soup)

msg = ''
for title in soup.find_all('h4', class_ = 'tit'):
    title_link = title.select('a')
    # print(title.link)
    article_url = title_link[0]['href']     # 0번째 이거 지워도 돌것 같대
    print(article_url)
    try:
        source_article = urllib.request.urlopen(article_url)
        soup = BeautifulSoup(source_article, 'lxml', from_encoding = 'utf-8')
        contents = soup.select('div.article_txt')
        print(contents)
        for imsi in contents:
            item = str(imsi.find_all(string = True))
            print(item)
            msg += item
        

    except Exception as e:
        pass

# print(msg)
okt = Okt()
nouns = okt.nouns(msg)

result = []
for imsi in nouns:
    if len(imsi) > 1:
        result.append(imsi)

print(result[:10])

count = Counter(result)
print(count)
tag = count.most_common(50)     # 상위 50개만 뽑아와

import pytagcloud
taglist = pytagcloud.make_tags(tag, maxsize = 100)
print(taglist)  # [{'color': (199, 87, 23), 'size': 137, 'tag': '여름'} 요런식으로 RGB까지 가져와
pytagcloud.create_tag_image(taglist, ' word.png', size = (1000,600),
                            background = (0,0,0), font_name = 'korean', rectangular=False)

# 이미지 읽기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()