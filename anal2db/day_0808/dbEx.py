import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import csv
from pandas import DataFrame, Series

# pandas 문제 7)
plt.rc('font', family= 'malgun gothic')     # 한글 깨짐 방지 코드 두줄
plt.rcParams['axes.unicode_minus']= False   # 한글 깨짐 방지 코드 두줄

#  a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.

#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성 ㅇㅇ

#      - DataFrame의 자료를 파일로 저장 ㅇㅇ

#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력 ㅇㅇ

#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))  ㅇㅇ

#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시

#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성 ㅇㅇ

config = {
    'host': '127.0.0.1',
    'user' : 'root',
    'password' : 'skfrnwl1@',
    'database' : 'mydb',
    'port' : 3306,
    'charset' : 'utf8'
}

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = '''
        select jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay
        from jikwon 
        inner join buser
        on jikwon.busernum = buser.buserno
            '''
    cursor.execute(sql)

#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성

    df1 = pd.DataFrame(cursor.fetchall(),
                      columns = ['jikwonno', 'jikwonname', 'busername', 
                      'jikwonjik', 'jikwongen', 'jikwonpay']                      
                      )
    # print(df1)

#      - DataFrame의 자료를 파일로 저장
    with open('jik_data.csv', mode = 'w', encoding = 'utf-8') as fobj:
        writer = csv.writer(fobj)
        for r in cursor:
            writer.writerow(r)
#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    buser_sum = df1.groupby(['busername'])['jikwonpay'].sum()
    buser_max = df1.groupby(['busername'])['jikwonpay'].max()
    buser_min = df1.groupby(['busername'])['jikwonpay'].min()
    print('각 부서별 연봉 총합 : ',buser_sum)
    print('각 부서별 연봉 최대 : ',buser_max)
    print('각 부서별 연봉 최소 : ',buser_min)
    
    #      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    ctab1 = pd.crosstab(df1['busername'],df1['jikwonjik'])
    print(ctab1)
    
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시

    sql = '''
        select gogekno, gogekname, gogektel
        from gogek
        inner join jikwon
        on jikwon.jikwonno = gogek.gogekdamsano
        '''
    
#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    buser_mean = df1.groupby(['busername'])['jikwonpay'].mean()
    plt.barh(buser_mean.index,buser_mean.values)
    plt.title('부서별 연봉 평균')
    plt.show()


except Exception as e:
    print('처리 오류 : ', e)
finally:
    conn.close()

#  b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.

config = {
    'host': '127.0.0.1',
    'user' : 'root',
    'password' : 'skfrnwl1@',
    'database' : 'mydb',
    'port' : 3306,
    'charset' : 'utf8'
}

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    
    sql = '''
        select jikwonname,busername, jikwonjik, jikwongen, jikwonpay
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
            '''
    cursor.execute(sql)
    df2 = pd.DataFrame(cursor.fetchall(),
                      columns = ['jikwonname','busername',
                                 'jikwonjik','jikwongen','jikwonypay']
                      )

#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
    df2.pivot_table(index = 'jikwongen', values = 'jikwonypay')
    # print(df2)
    man = df2[df2['jikwongen'] == '남']
    woman = df2[df2['jikwongen'] == '여']

    mean_manYpay = round(man['jikwonypay'].mean(),2)
    mean_womanYpay = round(woman['jikwonypay'].mean(),2)
    print('남성 직원 평균 연봉 : ', mean_manYpay)
    print('여성 직원 평균 연봉 : ', mean_womanYpay)
    
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
    labels = ['남성 직원 평균 연봉', '여성 직원 평균 연봉']
    values = [mean_manYpay,mean_womanYpay]
    plt.bar(labels, values)
    plt.title(config['database'])
    plt.ylabel('단위 : (만) 원')
    # plt.show()

#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
    ctab2 = pd.crosstab(df2['busername'],df2['jikwongen'])
    print(ctab2)
    print(df1.head())

except Exception as e:
    print('처리 오류 : ', e)
finally:
    conn.close()

 

#  c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.

#       조건 :  try ~ except MySQLdb.OperationalError as e:      사용

#      사번  직원명  부서명   직급  부서전화  성별

#      ...

#      인원수 : * 명