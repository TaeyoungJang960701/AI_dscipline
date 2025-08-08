# Local Database 연동 후 자료를 읽어 DataFrame에 저장
import sqlite3
import pandas as pd

sql = 'create table if not exists test(product varchar(10),maker varchar(10), weight real,price integer)'
conn = sqlite3.connect(':memory:')  # 'testdb' 연결 객체 만든거다
conn.execute(sql)
conn.commit()

data1 = ('mouse', 'samsung', 12.5, 5000)    # 이건 튜플이나 리스트로 정의해도 전혀 상관없다
# data1 = 'mouse', 'samsung', 12.5, 5000    이런 형태도 리스트로 읽어준대

# (4) 랑 (4,)를 보면 (4)는 튜플이 아니야 (4,) 이건 튜플이 맞아

conn.execute(sql)
conn.commit()

# 한 개씩 추가
stmt = 'insert into test values(?,?,?,?)'   # 시큐어 코딩 가이드라인에 저촉된다
data1 = ('mouse', 'samsung', 12.5, 5000)
conn.execute(stmt, data1)
data2 = ('mouse2', 'samsung', 15.5, 8000)
conn.execute(stmt, data2)

# 복수 개 추가
datas = [('mouse3', 'lg', 22.5, 15000), ('mouse4','lg',25.5,15500)]
conn.executemany(stmt,datas)    # 여러 개 추가할때는 매니를 붙여야돼

cursor = conn.execute('select * from test')
rows = cursor.fetchall()
# print(rows[0], '', rows[1], rows[0][0])
for a in rows:
    print(a)

# sql로 가져오는 방법 1
df = pd.DataFrame(rows, columns = ['product','maker','weight','price'])
print(df)
# print(df.to_html())     # 쟝고를 쓸 때는 이렇게 템플릿으로도 보낼 수 잇나봐
print('-' * 60)

# sql로 가져오는 방법 2
df2 = pd.read_sql('select * from test', conn)
print(df2)
print('-' * 60)

p_data = {
    'product' : ['연필','볼펜','지우개'],
    'maker' : ['동아','모나미','모나미'],
    'weight' : [1.5, 5.5, 10.0],
    'price' : [500, 1000, 1500]
}

frame = pd.DataFrame(p_data)
# print(frame)
frame.to_sql('test', conn, if_exists = 'append', index = False)    # 잇으면 추가해라 어펜드 해라 이얘기

df3 = pd.read_sql('select product, maker, price 가격, weight as 무게 from test', conn)  
# as 안쓰고 가격 해도 알아서 잘 읽어

print(df3)

cursor.close()
conn.close()

