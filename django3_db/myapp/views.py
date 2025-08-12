from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd

# Create your views here.
def indexFunc(request):
    return render(request,'index.html')

def dbshowFunc(request):
    dept = request.GET.get('dept',"").strip()
    # inner join
    sql = '''
        select j.jikwonno as 직원번호, j.jikwonname as 직원명,
        b.busername as 부서명, b.busertel as 부서전화,
        j.jikwonpay as 연봉,
        j.jikwonjik as 직급정보
        from jikwon j inner join buser b
        on j.busernum = b.buserno
    '''
    params = []     # params 리스트에 파라미터 값으로 %영업%처럼 %를 붙여 
                    # 부분검색 가능하도록 함
    if dept:
        sql += ' where b.busername like %s'
        params.append(f'%{dept}%')  # SQL 해킹방지 (시큐어 코딩)
    sql += ' order by j.jikwonno'

    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        print('cur.description : ', cur.description)
        cols = [c[0] for c in cur.description]  # 컬럼의 0번째인 이름만 cols에 저장
        print('cols : ', cols)

    df = pd.DataFrame(rows, columns = cols)
    # print(df.head(3)) 

    # join 결과로 html 생성
    if not df.empty:
        join_html = df[['직원번호','직원명','부서명','부서전화','연봉','직급정보']].to_html(index = False)
    else:
        join_html = '조회된 자료가 없습니다'

    # 직급별 연봉 통계표 (NaN -> 0 처리)
    if not df.empty:
        stats_df = (
            df.groupby('직급정보')['연봉']
                .agg(평균 = 'mean', 표준편차 = lambda x: x.std(ddof = 0), 인원수 = 'count')
                .round(2)
                .reset_index()
                .sort_values(by = '평균', ascending = False)
        )
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        stats_html = stats_df.to_html(index = False)
    else:
        stats_html = '통계 대상 자료가 없습니다'

    ctx_dict = {
        'dept' : escape(dept),  # 문자열에 특수문자가 있는 경우 HTML 엔티티로 치환해줌
                                # 예) escqpe('<script>alert(1)</script> -> '&lt;script&gt;...)
        'join_html' : join_html,
        'stats_html' : stats_html,
    }


    return render(request, 'dbshow.html',ctx_dict)