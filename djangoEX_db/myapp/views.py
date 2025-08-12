from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# pandas 문제 8)

# MariaDB에 저장된 jikwon, buser 테이블을 이용하여 아래의 문제에 답하시오.

# Django 모듈을 사용하여 결과를 클라이언트 브라우저로 출력하시오.
#    1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
#        : 부서번호, 직원명 순으로 오름 차순 정렬 
#    2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
#    3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
#    4) 성별, 직급별 빈도표를 출력하시오.
print('함수 호출됨')
# Create your views here.
def dbshowFunc(request):
    print('함수 호출됨')
    dept = request.GET.get('dept',"").strip()
    # inner join
#    1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
    sql = '''
        select j.jikwonno as 사번, j.jikwonname as 직원명,
        b.busername as 부서명, j.jikwonpay as 연봉, b.buserno as 부서번호,
        j.jikwonjik as 직급, j.jikwonibsail as 입사일, j.jikwongen as 직원성별
        from jikwon j inner join buser b
        on j.busernum = b.buserno 
    '''
#        : 부서번호, 직원명 순으로 오름 차순 정렬
    params = []
    if dept:
        sql += ' where b.busername like %s'
        params.append(f'%{dept}%')      # f 스트링으로 쓰는 이유는 print('cons',cons) 이렇게 변수를 받으면 해킹당하기 쉽대
    sql += ' order by b.buserno, j.jikwonname '
    
    # working_year = (datetime() - ' select * from j.jikwonibsail') // 365
    # 이런 방식으로 돌아가는게 아니래  

    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        print('cur.description : ', cur.description)
        cols = [c[0] for c in cur.description]
        print(f'cols : %{cols}%')

    df = pd.DataFrame(rows, columns = cols)
    df['입사일'] = pd.to_datetime(df['입사일'])
    df['근무년수'] = ((pd.Timestamp.now() - df['입사일']).dt.days // 365)
    print(df.head(3))

#    2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.    
    join_html = df.to_html(index = False)
    stats_df = (
        df.groupby(['부서명','직급'])['연봉']
        .agg(연봉합 = 'sum', 연봉평균 = 'mean')
        .round(2)
        .reset_index()
        .sort_values(by = '연봉평균', ascending = False)
    )
    stats_html = stats_df.to_html(index = False)
#    3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
    graph_df = (df.groupby(['부서명'])['연봉']
                .agg(평균 = 'mean', 총합 = 'sum')
                .round(2)
                .reset_index()   
    )

    plt.bar(graph_df['부서명'],graph_df['총합'])
    plt.xlabel('부서명')
    plt.ylabel('연봉 총합계')
    plt.title('부서별 연봉 총합계')
    plt.savefig('static/img/graph.png')
    plt.show()

#    4) 성별, 직급별 빈도표를 출력하시오.
    table_df = pd.crosstab(df['직원성별'],df['직급'])
    table_html = table_df.to_html()

    ctx_dict = {
        'dept' : escape(dept),
        'join_html' : join_html,
        'stats_html' : stats_html,
        'table_html' : table_html,
    }
    return render(request, 'dbshow.html', ctx_dict)


# 참고 : 원격 table 구조 얻기 > python manage.py inspectdb > a.py