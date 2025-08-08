# matplotlib는 플로팅 모듈. 다양한 그래프 지원, 함수 지원

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic') # 한글 깨짐 방지 
plt.rcParams['axes.unicode_minus'] = False # 한글 깨짐 방지 
'''
x = ['서울','인천','수원']
y = [5,4,7]
plt.xlim([-1,3])
plt.ylim([0,10])
plt.plot(x, y)
plt.yticks(list(range(0,10,3)))
plt.show()  # show는 맷플랏립의 메서드임 피규어를 띄워주고 일시정지한대 
            # 그래서 밑의 ㅇㅋ가 실행이 안되는거야
# 서울 인천 수원의 좌표는 (x,y) = (0,5), (1,4), (2,7) 이래

# jupiter notebook에서는 매직 명령어를 사용한대 '%matplotlib inline' 하면 show() 없어도 된대
# print('ok')

data = np.arange(1,11,2) # 1에서 11(-1)인 10까지 2칸 띄워서. 2칸이 step 인가봐
print(data)
plt.plot(data)
x = [0,1,2,3,4]
for a,b in zip(x,data):
    plt.text(a,b,str(b))

plt.show()

plt.plot(data)
plt.plot(data,data,'r')
for a,b in zip(x,data):
    plt.text(a,b,str(b))
plt.show()
'''

# # sin 곡선
# x = np.arange(10)
# y = np.sin(x)
# print(x,y)
# # plt.plot(x,y)
# # plt.plot(x,y, 'r+') 스타일을 정해주는게 가능하대 이거 근데 완전 매트랩 아니냐 
# # 'b'lue색의 o로 점을 찍는다
# plt.plot(x,y,'go--', linewidth = 2, markersize = 12)
# # color = 'b' => c = 'b', lw = 2, marker = 'o', ms = 12
# # - (solid line), --(dashed line) ...
# plt.show()

# 홀드 명령 : 하나의 영역에 두 개 이상의 그래프 표시
x = np.arange(0,3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# plt.figure(figsize = (10,5))
# plt.plot(x,y_sin,'r')
# # plt.hold()
# plt.plot(x,y_cos,'b') # plt.scatter(x,y_cos,'b') 이렇게 하는것도 가능해
# plt.xlabel('x축')
# plt.ylabel('y축')
# plt.title('제목')
# plt.legend(['sine','cosine'])
# plt.show()

# subplot : Figure를 여러 개 그림
"""plt.subplot(2,1,1)
plt.plot(x, y_sin)
plt.title('sine')
plt.subplot(2,1,2)
plt.plot(x, y_cos)
plt.title('cosine')
plt.show()"""

print()
irum = ['a','b','c','d','e']
kor = [80,50,70,70,90]
eng = [60,70,80,70,60]
plt.plot(irum,kor,'ro-')
plt.plot(irum,eng,'gs-')
plt.ylim([0,100])
plt.legend(['국어','영어'],loc = 2) # loc로 legend 위치를 지정할때는 1시방향에서 반시계방향으로 돈다
plt.grid(True)
fig = plt.gcf()
plt.show()
fig.savefig('result.png')

from matplotlib.pyplot import imread

img = imread('result.png')
plt.imshow(img)
plt.show()