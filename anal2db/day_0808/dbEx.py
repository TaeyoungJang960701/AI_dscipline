import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import csv
from pandas import DataFrame, Series
import os

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

# 비밀번호 설정을 다르게 했습니다 집에서는 비밀번호를 1234로 했어요
# 'password' : 'skfrnwl1@',

# config = {
#     'host': '127.0.0.1',
#     'user' : 'root',
#     'password' : '1234',
#     'database' : 'mydb',
#     'port' : 3306,
#     'charset' : 'utf8'
# } 이거는 피클 쓸때 필요 없는거래

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

    # 담당 고객이 없는 상황에 맞게 데이터베이스 가져오는방식에 ifnull을 사용
    sql = '''
        select jikwonno, jikwonname,
            ifnull(gogekno, '담당 고객 X') as gogekno,
            ifnull(gogekname, '담당 고객 X') as gogekname,
            ifnull(gogektel, '담당 고객 X') as gogektel
        from jikwon
        left join gogek
        on jikwon.jikwonno = gogek.gogekdamsano
        order by jikwon.jikwonno
    '''
    cursor.execute(sql)
    rows = cursor.fetchall()
    df_customers = pd.DataFrame(rows, columns=['사번','직원명','고객번호','고객명','고객전화'])
    print('고객 명단 :\n', df_customers)

    
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
    'password' : '1234@',
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
    print(df2)
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


config = {
    'host': '127.0.0.1',
    'user' : 'root',
    'password' : 'skfrnwl1@',
    'database' : 'mydb',
    'port' : 3306,
    'charset' : 'utf8'
}

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    # 사번, 직원명 입력 받기
    input_jikwonno = input('사번 입력: ').strip()
    input_jikwonname = input('직원명 입력: ').strip()

    # 로그인 검증용 SQL
    sql = '''
        select jikwon.jikwonno, jikwon.jikwonname, buser.busername, jikwon.jikwonjik,
               buser.busertel, jikwon.jikwongen
        from jikwon
        inner join buser on jikwon.busernum = buser.buserno
        where jikwon.jikwonno = %s and jikwon.jikwonname = %s
    '''
    cursor.execute(sql, (input_jikwonno, input_jikwonname))
    result = cursor.fetchall()

    if not result:
        print('로그인 실패: 사번 또는 직원명이 일치하지 않습니다.')
        sys.exit()

    # 로그인 성공 시 정보 출력
    print('사번\t직원명\t부서명\t직급\t부서전화\t성별')
    for row in result:
        print(f'{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}')

    # 로그인 성공 인원 수 출력 (같은 사번+이름으로 여러 레코드 있을 수 있으므로)
    print(f'인원수 : {len(result)} 명')

except MySQLdb.OperationalError as e:
    print('DB 연결 또는 쿼리 실행 오류:', e)
except Exception as e:
    print('처리 오류:', e)
finally:
    if 'conn' in globals():
        conn.close()
