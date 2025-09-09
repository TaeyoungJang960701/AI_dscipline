# iris dataset을 이용한 계층적 군집분석
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_df.head(2))
print(iris_df.loc[0:2,['sepal length (cm)','sepal width (cm)']])

from scipy.spatial.distance import pdist,squareform
# 1) 거리 계산에 사용할 부분 데이터 선택
dis_vec=pdist(iris_df.loc[0:4,['sepal length (cm)','sepal width (cm)']],metric='euclidean')     #   인덱스 0~4(총 5개 샘플)의 'sepal length'와 'sepal width' 두 특성만 사용
print('dis_vec : ',dis_vec)
# dis_vec :  [0.53851648 0.5        0.64031242 0.14142136 0.28284271 0.31622777 0.60827625 0.14142136 0.5        0.64031242]
print()

# 2) squareform으로 1차원 거리 벡터 → 정사각형 거리 행렬(대칭, 대각선 0) 변환
row_dist=pd.DataFrame(squareform(dis_vec))      #    보기 편하도록 DataFrame으로 감싸고, 인덱스/열 이름을 원래 행 인덱스로 지정
print('row_dist : ',row_dist)

from scipy.cluster.hierarchy import linkage,dendrogram
# 1) 링크지(linkage) 행렬 계산
# - dis_vec: pdist(...)로 만든 'condensed' 1차원 거리 벡터 (길이 = n*(n-1)/2)
# - method='complete' : 두 군집 사이의 '최대' 쌍거리(complete linkage)를 기준으로 병합
row_clusters=linkage(dis_vec,method='complete')     # ward, average...
print('row_cluster : ',row_clusters)
df=pd.DataFrame(row_clusters,columns=['id1','id2','거리','멤버수'])
print(df)

# 2) 덴드로그램(dendrogram)으로 계층적 군집 구조 시각화
# - labels=... 를 지정하면 리프(말단)에 원하는 라벨을 표시할 수 있습니다.
row_den=dendrogram(row_clusters)
plt.tight_layout()
plt.ylabel('dist')
plt.show()

print()
from sklearn.cluster import AgglomerativeClustering
# 계층적 군집(병합적) 모델 설정
# - n_clusters=2     : 원하는 최종 군집 개수(2개)
# - metric='euclidean': 데이터 포인트 간 거리 계산 방식(유클리드 거리)
# - linkage='complete': 두 군집 간 거리로 '최대 쌍거리(complete)' 사용
ac=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='complete')

# 군집 대상으로 사용할 특징 행렬 X 선택 (아이리스 0~4행, 2개 특징)
x=iris_df.loc[0:4,['sepal length (cm)','sepal width (cm)']]
labels=ac.fit_predict(x)
print('클러스터 분류 결과 : ',labels)

plt.hist(labels)
plt.grid()
plt.show()