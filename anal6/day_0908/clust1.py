# 클러스터링 기법 중 계층적 군집화 이해

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun gothic')

np.random.seed(123)  # 같은 seed 값(123)을 주면, np.random.*로 생성되는 난수가 매 실행마다 동일해집니다.
var=['x','y']
labels=['점0','점1','점2','점3','점4']
x=np.random.random_sample([5,2])*10
df=pd.DataFrame(x,columns=var,index=labels)
print(df)

# plt.scatter(x[:,0],x[:,1],c='blue',marker='o',s=50)
# plt.grid(True)
# plt.show()

from scipy.spatial.distance import pdist,squareform       # 클러스트링 되는 모습을 보기위해 사용
# pdist : 배열에 있는 값을 이용해 각 요소들의 거리를 계산
# squareform : 거리벡터를 사각형 형식으로 변환하는 역할
dist_vec=pdist(df,metric='euclidean')
print('dist_vec : ',dist_vec)
# 어떤점들과의 거리차이인지 모름
# dist_vec :  [5.3931329  1.38884785 4.89671004 2.40182631 5.09027885 7.6564396
#  2.99834352 3.69830057 2.40541571 5.79234641]

row_dist=pd.DataFrame(squareform(dist_vec),columns=labels,index=labels)     # 이 과정을 통해서 어떤 점과 어떤점의 사이 거리를 확인할수 있음. 
print(row_dist)
#           점0       점1       점2       점3       점4
# 점0  0.000000  5.393133  1.388848  4.896710  2.401826
# 점1  5.393133  0.000000  5.090279  7.656440  2.998344
# 점2  1.388848  5.090279  0.000000  3.698301  2.405416
# 점3  4.896710  7.656440  3.698301  0.000000  5.792346
# 점4  2.401826  2.998344  2.405416  5.792346  0.000000

# 응집형 : 자료 하나하나를 군집으로 보고 가까운 군집끼리 연결해 나가는 방법. 상향식
# 분리형 : 전체자료를 하나의 군집으로 보고 분리해 나가는 방법. 하향식

# linkage : 응집형 계층적 군집을 수행
from scipy.cluster.hierarchy import linkage
row_clusters=linkage(dist_vec,method='ward')

df=pd.DataFrame(row_clusters,columns=['클러스터id_1','클러스터id_1','거리','클러스터 멤버수'])
print(df)

# linkage의 결과로 덴드로그램 작성(어떻게 응집되었는지를 보여주는 것이 덴드로그램 그래프)
from scipy.cluster.hierarchy import dendrogram
row_dendr=dendrogram(row_clusters,labels=labels)
plt.tight_layout()
plt.ylabel('유클리드 거리')
plt.show()



