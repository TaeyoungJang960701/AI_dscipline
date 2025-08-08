import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = range(10)
"""
# fugure 구성 방법
# 1) matplotlib 스타일의 인터페이스
plt.figure()
plt.subplot(2,1,1) # row, col, panel number 그니까 행, 열, 순서야
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x,np.cos(x))
plt.show()

# 2) 객체 지향 인터페이스 - - - - - - - 결국에 첫번째 방법이랑 똑같은 그림이긴 해
fig, ax = plt.subplots(nrow = 2, ncols = 1)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()
"""

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(np.random.randn(10), bins = 3, alpha = 0.9) # randn이 정규분포를 따르는 수래
ax2.plot(np.random.randn(10))
plt.show()

# bar
data = [50,80, 100, 70, 90]
# plt.bar(range(len(data)), data)
plt.bar(range(len(data)), data) # 이건 세로막대
plt.show()
"""
loss = np.random.randn(len(data))
plt.barh(range(len(data)),data, xerr = loss, alpha = 0.7) # 이건 가로막대 horizon
plt.show()
"""

# pie 원형 파이컷 그래프

plt.pie(data, explode = (0, 0.1, 0,0,0), colors = ['yellow','red','blue'])
plt.show()

# boxplot 엄청 중요한 녀석이래
# 사분위 등에 의한 데이터 분포 확인에 존내 효과적
plt.boxplot(data)
plt.show()

# 버블 차트
n = 30
np.random.seed(0)
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale  = np.pi * 15 * ( np.random.rand(n)) **2
plt.scatter(x, y, c = color, s = scale)
plt.show()

fdata = pd.DataFrame(np.random.randn(1000,4), 
                     index =pd.date_range('1/1/2000', periods = 1000), columns = list('ABCD'))
fdata = fdata.cumsum()
print(fdata.head(3))
plt.plot(fdata)
plt.show()

# pandas 가 지원하는 plot
fdata.plot()
fdata.plot(kind = 'bar')
fdata.plot(kind = 'box')
plt.xlabel('time')
plt.ylabel('data')
plt.show()