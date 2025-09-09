# K-means 알고리즘이란?
# 머신러닝 비지도학습에 속하는 K-means 알고리즘은 쉽게 말해 데이터를 K개의 군집(Cluster)으로 묶는(Clusting) 알고리즘.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs     # clust 연습용 데이터셋
# 2차원 상에 3개의 군집으로 구성된 점 150개를 생성
# - n_samples=150   : 총 샘플 수
# - n_features=2    : 특징(좌표) 차원 수 = 2D (x, y)
# - centers=3       : 실제 군집 개수
# - cluster_std=0.5 : 군집의 퍼짐(표준편차). 작을수록 더 조밀
# - shuffle=True    : 생성된 샘플 섞기
# - random_state=0  : 난수 고정(재현성)
# - 반환값: (데이터, 정답라벨). 여기서는 정답라벨은 사용하지 않아 '_'에 버림
x,_=make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0)
print(x[:5],x.shape)

# 산점도(Scatter plot)로 생성된 점들을 확인
# - x[:,0] : 첫 번째 특징(가로축), x[:,1] : 두 번째 특징(세로축)
# - c='gray' : 모든 점을 회색으로 동일하게 표시
# - marker='o' : 원형 마커, s=50 : 점 크기
# plt.scatter(x[:,0],x[:,1],c='gray',marker='o',s=50)
# plt.grid()
# plt.show()

from sklearn.cluster import KMeans
init_centroid='random'       # 초기 클러스터 중심을 임의로 선택  
# init_centroid='k-means++'  # 초기 중심을 더 안정적으로 뽑는 방식(일반적으로 수렴/성능이 더 좋음)[기본]

kmodel=KMeans(n_clusters=3,init=init_centroid,random_state=0).fit(x)
pred=kmodel.fit_predict(x)      # 모델을 학습시키기
# print('pred : ',pred)         
# print(x[pred == 0])           # pred == 0 인 샘플(클러스터 라벨 0에 속한 점들)만 출력

# KMeans로 학습된 군집 중심(centroid) 좌표 출력
# - 형태: (n_clusters, n_features). 여기서는 (3, 2)  → 2차원 평면상의 3개 중심점
print('centroid : ',kmodel.cluster_centers_)

# 각 군집에 속한 점들을 서로 다른 색/마커로 시각화
# - pred == k : 예측 라벨이 k(0,1,2)인 샘플만 선택
# - x[mask, 0] : 선택된 점들의 x좌표, x[mask, 1] : y좌표
plt.scatter(x[pred ==0,0],x[pred==0,1],c='red',marker='o',s=50,label='cluster1')        # 군집 0
plt.scatter(x[pred ==1,0],x[pred==1,1],c='yellow',marker='s',s=50,label='cluster2')     # 군집 1
plt.scatter(x[pred ==2,0],x[pred==2,1],c='blue',marker='v',s=50,label='cluster3')       # 군집 2

# 군집 중심(centroid) 위치 표시
# - kmodel.cluster_centers_[:, 0] : 중심들의 x좌표
# - kmodel.cluster_centers_[:, 1] : 중심들의 y좌표
plt.scatter(kmodel.cluster_centers_[:,0],kmodel.cluster_centers_[:,1],c='black',marker='+',s=80,label='center')


plt.legend()
plt.grid()
plt.show()

# 가장 합리적인 클러스터 중심점 갯수 구하기
# 방법1) elbow기법 - 클러스터 간 SSE의 차이를 이용해 최적의 클러스터 수 반환
def elbowFunc(x):
    sse=[]
    for i in range(1,11):
        km=KMeans(n_clusters=i,init='k-means++',random_state=0)
        km.fit(x)
        sse.append(km.inertia_)
    plt.plot(range(1,11),sse,marker='o')
    plt.xlabel('count cluster')
    plt.ylabel('sse')
    plt.show()

elbowFunc(x)

# 방법2)실루엣(silhouette) 기법
'''
실루엣(silhouette) 기법
  클러스터링의 품질을 정량적으로 계산해 주는 방법이다.
  클러스터의 개수가 최적화되어 있으면 실루엣 계수의 값은 1에 가까운 값이 된다.
  실루엣 기법은 k-means 클러스터링 기법 이외에 다른 클러스터링에도 적용이 가능하다
'''

import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm

def plotSilhouette(x, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장
    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기
        c_sil_value = sil_val[pred == c]
        c_sil_value.sort()
        y_ax_upper += len(c_sil_value)

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_sil_value)

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 개수')
    plt.show() 

plotSilhouette(x,pred)