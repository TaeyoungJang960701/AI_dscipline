# # 웹문서 읽기
# # 위키백과 문서 읽기 - 이순신 자료
# import urllib.request as req
# from bs4 import BeautifulSoup
# import urllib



# # 위키백과 문서 읽기 - 이순신 자료
# # url = 'https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0'
# # wiki = req.urlopen(url)
# # print(wiki)

# # soup = BeautifulSoup(wiki, 'html.parser')
# # print(soup.select('#mw-content-text > div.mw-parser-output > p'))

# import csv;
# import re
# import pandas as pd
# import requests

# # 네이버 제공 코스피 정보 읽기 - DataFrame에 담아 지지고볶고...
# url_template = 'https://finance.naver.com/sise/nxt_sise_market_sum.naver?&page={}'
# csv_fname = '네이버 코스피.csv'
# with open(csv_fname, mode = 'w', encoding = 'utf-8', newline = '') as f:
#     writer = csv.writer(f)
#     # 제목
#     headers = 'N	종목명	현재가	전일비	등락률	액면가	시가총액	상장주식수	외국인비율	거래량	PER	ROE	토론'.split(' ')
#     writer.writerow(headers)


#     for page in range(1,3):
#         url = url_template.format(page)
#         # print(url)
#         res = requests.get(url)
#         res.raise_for_status() # 실패하면 작업 중지하도록 하는 코드
#         soup = BeautifulSoup(res.text, 'html.parser')
#         # rows = soup.find('table', attrs = {'class':'type_2'}).find('tbody').find_all('tr')
#         rows = soup.select('table.type_2 tbody tr')
#         # print(rows)

#         for row in rows:
#             cols = row.find_all('td')
#             if len(cols) < len(headers):
#                 print(f'[스킵됨] 열(rows) 수 부족:{len(cols)}개')
#                 continue
#             row_data = [re.sub(r'[\n\t] + ', '', col.get_text().strip()) for col in cols]
            
#             writer.writerow(row_data)
        
# print('csv 성공')
# df = pd.read_csv(csv_fname, dtype = str, index_col = False)
# print(df.head(3))
# print(df.columns.tolist()) # 칼럼(column)명 보기
# print(df.info())

# numeric_cols = ['현재가','등락률','액면가','']

# # 전일비 전용 전처리 함수
# def clean_change_direction(val):
#     if pd.isna(val):
#         return pd.NA
#     val = str(val)
#     val = val.replace(',','').replace('상승','+').replace('하락','-')
#     val = re.sub(r'[^\d\.\-\+]','',val) # 숫자/기호 외 문자 제거
#     try:
#         return float(val)
#     except ValueError:
#         return pd.NA
    
# df['전일비'] = df['전일비'].apply(clean_change_direction)
# print(df.head(3))


# ----------------------------------------여기 아래로는 지피티가 고쳐준거야
import requests
from bs4 import BeautifulSoup
import csv
import re
import pandas as pd

# 네이버 코스피 시가총액 페이지
url_template = 'https://finance.naver.com/sise/sise_market_sum.naver?&page={}'
csv_fname = '네이버 코스피.csv'

# 저장할 열 (웹사이트 구조와 일치하는 것만)
headers = ['종목명', '현재가', '전일비', '등락률', '액면가', '시가총액', 
           '상장주식수', '외국인비율', '거래량', 'PER', 'ROE']

with open(csv_fname, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    for page in range(1, 3):
        url = url_template.format(page)
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.select('table.type_2 tbody tr')

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < len(headers):
                continue
            row_data = [col.get_text(strip=True).replace(',', '') for col in cols]
            writer.writerow(row_data[:len(headers)])  # 딱 맞춰 자르기

print('CSV 저장 완료')

# 읽기
df = pd.read_csv(csv_fname, dtype=str)
print(df.head(3))
print(df.columns.tolist())
print(df.info())

numeric_cols = ['현재가', '전일비', '등락률', '액면가', '시가총액', 
           '상장주식수', '외국인비율', '거래량', 'PER', 'ROE']

# 전일비 전용 전처리 함수
def clean_change_direction(val):
    if pd.isna(val):
        return pd.NA
    val = str(val).replace('상승', '+').replace('하락', '-').replace(',', '')
    val = re.sub(r'[^\d\.\-\+]', '', val)
    try:
        return float(val)
    except ValueError:
        return pd.NA

df['전일비'] = df['전일비'].apply(clean_change_direction)
print(df[['종목명', '전일비']].head(3))

# 일반 숫자형 컬럼(column) 전처리
def clean_numeric_column(series):
    return(
        series.astype(str)
            .str.replace(',','',regex = False)
            .str.replace('%','',regex = False)
            .replace(['','-','N/A','nan'],pd.NA)
            .apply(lambda x:pd.to_numeric(x, errors = 'coerce'))
    )

for col in numeric_cols:
    df[col] = clean_numeric_column(df[col])
print('숫자 칼럼(column) 일괄 처리 후')
print(df.head(2))

print('-' * 60)
print(df.describe())
print(df[['종목명','현재가','전일비']].head(5))
print('시가 총액 top 5')
top5 = df.dropna(subset = ['시가총액']).sort_values(by = '시가총액',ascending = False).head()
print(top5[['종목명','시가총액']])