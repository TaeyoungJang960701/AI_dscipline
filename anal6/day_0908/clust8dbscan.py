# 밀도기반 클러스터링(dbscan)
# 밀도 기반 클러스터링 비모수적 알고리즘이다. 일부 공간에 있는 점의 경우, 
# 서로 밀접하게 밀집된 점(인근 이웃이 많은 점)을 그룹화하여 저밀도 지역(가장 가까운 이웃이 너무 멀리 떨어져 있음)에 혼자 있는 이상점으로 표시한다. 
# DBSCAN은 가장 일반적이고 가장 많이 인용되는 클러스터링 알고리즘 중 하나이다

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans,DBSCAN

# 1) 반달 모양 합성 데이터 생성
# - n_samples=200 : 샘플 200개
# - noise=0.05    : 노이즈(산포) 정도
# - random_state=0: 재현성(난수 고정)
x,y=make_moons(n_samples=200,noise=0.05,random_state=0)
print(x[:10])
print('실제 군집 id : ',y[:10])

# plt.scatter(x[:,0],x[:,1])
# plt.show()

# KMeans로 군집화
# - n_clusters=2 : 두 개의 클러스터로 나눔(라벨 번호는 임의)
km=KMeans(n_clusters=2,random_state=0)
pred1=km.fit_predict(x)
print('예측 군집 id : ',pred1[:10])

# 시각화
def plotResultFunc(x,pr):
    plt.scatter(x[pr==0,0],x[pr==0,1],c='blue',marker='o',s=40,label='cluster-1')       # pr==0인 점(군집 0)을 파란 원으로 표시
    plt.scatter(x[pr==1,0],x[pr==1,1],c='red',marker='s',s=40,label='cluster-2')        # pr==1인 점(군집 1)을 빨간 사각형으로 표시
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='+',s=50,label='centroid')       # KMeans로 얻은 군집 중심(centroid) 표시(+ 마커)
    plt.legend()
    plt.show()

plotResultFunc(x,pred1)     # 데이터의 중심값이 이상하여 원하는 결과가 출력 되지 않음 -> 밀도(dbscan사용해야함)

print()
# dbscan으로 균집화
# DBSCAN(밀도 기반 군집)
# - eps=0.2       : 한 점이 '이웃'으로 간주되는 최대 거리(반경)
# - min_samples=5 : 핵심점(core point) 판정에 필요한 최소 이웃 수(자기 자신 포함)
#   → 반경 eps 안에 min_samples개 이상 점이 모여 있으면 '밀집 영역'으로 보고 군집을 확장
dm=DBSCAN(eps=0.2,min_samples=5)

# 군집 라벨 예측
# - 각 샘플의 라벨을 반환 (0,1,2,...) ; 잡음(아무 군집에도 속하지 않음)은 -1로 표시
pred2=dm.fit_predict(x)

plotResultFunc(x,pred2) 

# 군집화 : 고객 세분화,예상치 탐색, 추천 시스템...등의 효과적