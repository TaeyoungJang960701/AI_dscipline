# pip install konlpy

from konlpy.tag import Okt, Kkma, Komoran
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# corpus(말뭉치) : 자연어 처리를 목적으로 수집된 문장 집단
text = '나는 오늘 아침에 강남에 갔다. 가는 길에 빵집이 보여 너무 먹고 싶었다.'

# 형태소 : 단어의 최소 단위
print('Okt -----')
okt = Okt()
print('형태소 : ', okt.morphs(text))
print('-' * 70)
print('품사 태깅 : ', okt.pos(text))
print('-' * 70)
print('품사 태깅(어간 포함) : ', okt.pos(text, stem = True))    # 원형(어근)으로 출력.
print('-' * 70)
# 그래요 -> 그렇다 이런식으로, 기본형으로 바꿔주나봐
print('명사 추출 : ', okt.nouns(text))
print('-' * 70)

print('Kkma -----')
kkma = Kkma()
print('형태소 : ', kkma.morphs(text))
print('-' * 70)
print('품사 태깅 : ', kkma.pos(text))
print('-' * 70)
print('품사 태깅(어간 포함) : ', kkma.pos(text))    # 원형(어근)으로 출력.
print('-' * 70)
# 그래요 -> 그렇다 이런식으로, 기본형으로 바꿔주나봐
print('명사 추출 : ', kkma.nouns(text))
print('-' * 70)

print('Komoran -----')
komoran = Komoran()
print('형태소 : ', komoran.morphs(text))
print('-' * 70)
print('품사 태깅 : ', komoran.pos(text))
print('-' * 70)
print('품사 태깅(어간 포함) : ', komoran.pos(text))   # 원형(어근)으로 출력.
print('-' * 70)
# 그래요 -> 그렇다 이런식으로, 기본형으로 바꿔주나봐
print('명사 추출 : ', komoran.nouns(text))
print('-' * 70)

print('-------워드클라우드--------')
# pip install wordcloud
text2 = '나는 오늘 아침에 강남에 갔다. 가는 길에 빵집이 보여 너무 먹고 싶었다. 빵이 특히 강남에 있는'
nouns = okt.nouns(text2)
words = ' '.join(nouns)
print('words : ', words)

wc = WordCloud(font_path = 'malgun.ttf', width = 400, height = 300, background_color = 'white')
print('-' * 70)
cloud = wc.generate(words)

plt.imshow(cloud, interpolation = 'bilinear')   # 이미지 부드럽게 출력
plt.axis('off')     # 워드클라우드니까 축이름은 꺼둬
plt.show()