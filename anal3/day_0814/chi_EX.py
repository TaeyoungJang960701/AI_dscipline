# * 카이제곱 검정

# 카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
#   예제파일 : cleanDescriptive.csv
#   칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
#   조건 :  level, pass에 대해 NA가 있는 행은 제외한다.

import pandas as pd
import scipy.stats as stats
import pickle
import sys
import MySQLdb

# 귀무가설(H0) : 부모와 자식의 학력에는 상관관계가 없다. (독립적이다)
# 대립가설(H1) : 부모와 자식의 학력은 관계가 있다. (종속적이다)
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv')
data = data[data['level'] != 'Na']
data = data[data['pass'] != 'Na']

parents_level = data['level']
offspring_level = data['pass']

print(parents_level, offspring_level)
print('-' * 100)

ctab = pd.crosstab(columns = parents_level, index = offspring_level)    
# 위 ctab에서 뺀 아규먼트 normalize는 이건 "비율화" 하는거야 온전한 수를 볼 수 없게돼

print(ctab)
print('-' * 100)

chi2, p, dof, expected = stats.chi2_contingency(ctab)
msg = '부모와 자식의 학력 상관관계 - \nchi2 : {}, p-value : {}, dof : {}, expected(기대값) : {}'
print(msg.format(chi2,p,dof,expected))
print('-' * 100)
# 부모와 자식의 학력 상관관계 -
# chi2 : 2.766951202595669, p-value : 0.25070568406521354, 
# dof : 2, expected(기대값) :   [[53.4 49.2 32.4]
#                               [35.6 32.8 21.6]]
# 결론
# p-value가 유의수준 0.05를 한참 넘긴다.
# 따라서 대립가설은 기각하고 귀무가설을 채택한다.
# 부모와 자식의 학력 수준은 관련이 없다. 부모 학력이 좋다고 자식 학력이 좋지 않다.


# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 

# 그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
#   예제파일 : MariaDB의 jikwon table 
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#   조건 : NA가 있는 행은 제외한다. 

# 귀무가설(H0) : 각 직원들의 직급과 연봉의 높고 낮음에는 관련성이 없다. (직급에 따라 연봉은 차이가 없다)
# 대립가설(H1) : 각 직원들의 직급과 연봉의 높고 낮음에는 관련이 있다. (직급에 따라 연봉이 매겨진다)

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('로드 오류 : ',e)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = """
        select jikwonjik as 직급, jikwonpay as 연봉 from jikwon
    """

    cursor.execute(sql)

    data2 = pd.DataFrame(cursor.fetchall(), 
                         columns = ['직급','연봉']
                         )
    # print(data2)
    # data2 = data[data['직급'] != 'NA']
    # data2 = data[data['연봉'] != 'NA']
    print(data2)
    print('-' * 100)

    # ctab_j = pd.crosstab(index = ['직급'], columns = ['연봉'])
    # print(ctab_j)
    # print('-' * 100)
    # jikpay = data2.groupby(['직급'])['연봉'].sum()
    # print(jikpay)
    #   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
    jik = ['이사','부장','과장','대리','사원']
    jik_bound = [1,2,3,4,5]
    # jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
    pay_bound = [1000,3000,5000,7000,1000000]
    pay = [1,2,3,4]
    
    data2['직급등급'] = data2['직급']
    data2['연봉등급'] = data2['연봉']
    # print(data2)
    data2['직급등급'] = data2['직급'].map({'이사':1, '부장':2, '과장':3,'대리':4,'사원':5})
    # print(data2)
    data2['연봉등급'] = pd.cut(data2['연봉등급'], bins = pay_bound, labels = pay)
    # print(data2)
    ctab2 = pd.crosstab(index = data2['직급등급'], columns = data2['연봉등급'])
    print(ctab2)
    print('-' * 100)
    chi2, p, dof, expected = stats.chi2_contingency(ctab2)
    msg = '직원들 직급과 연봉의 상관관계 ----\nchi2 : {}, p-value : {}, dof : {}, expected : {}'
    print(msg.format(chi2, p, dof, expected))
    
    # chi2 : 36.27472527472528, p-value : 0.00029263428943485575, 
    # dof : 12, expected : [[0.16666667 0.4        0.16666667 0.26666667]
    # 결론 
    # p값이 약 0.0003으로 유의수치 0.05 보다 낮으므로 귀무가설을 기각아서 대립가설을 채택한다.
    # 직원들의 연봉은 직급에 따라 매겨진다.
    
except Exception as e:
    print('처리 오류 : ',e)
finally:
    conn.close()