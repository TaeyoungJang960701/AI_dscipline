# 가상의 데이터로 '쇼핑몰 고객 세분화(집단화)' 연습
# KMeans 군집화
# 고객수 , 연간지출액, 방문수 ...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(0)   # 난수 시드 고정 → 실행할 때마다 동일한 값(재현성)
n_customers=200     # 고객 수
annual_spending=np.random.normal(50000,15000,n_customers)   # 연간 지출액(원화 가정): 평균 50,000 / 표준편차 15,000의 정규분포에서 샘플 생성
monthly_visits=np.random.normal(5,2,n_customers)     # 월 방문 횟수: 평균 5회 / 표준편차 2회의 정규분포에서 샘플 생성
# print(annual_spending)
# print(monthly_visits[:5])
# np.clip: 값의 하한/상한을 잘라내기
# - 지출액/방문수는 음수가 될 수 없으므로 하한을 0으로 고정
annual_spending=np.clip(annual_spending,0,None)
monthly_visits=np.clip(monthly_visits,0,None)
print(annual_spending[:5])
# [76460.78518951 56002.35812551 64681.06976159 83613.39798802
#  78013.36985225]
print(monthly_visits[:5])
# [4.26163632 4.52124164 7.19931919 6.31052746 6.28026305]

# 분석/모델링에 쓰기 좋게 DataFrame으로 정리
data=pd.DataFrame({
    'annual_spending':annual_spending,
    'monthly_visits':monthly_visits
})
print(data.head(2))
#    annual_spending  monthly_visits
# 0     76460.785190        4.261636
# 1     56002.358126        4.521242

# 산포도
plt.scatter(data['annual_spending'],data['monthly_visits'])   # x: 연간 지출액, y: 월 방문수
plt.xlabel('annual_spending')                                 # x축 라벨
plt.ylabel('monthly_visits')                                  # y축 라벨
plt.show()

# KMeans 군집화
Kmeans=KMeans(n_clusters=3, random_state=0)  # 군집 개수=3, 난수 고정으로 재현성 확보
clusters=Kmeans.fit_predict(data)            # 데이터로 학습(fit) + 각 샘플의 군집 라벨 예측(predict)

# 군집결과 시각화
data['cluster']=clusters
print(data.head(5))

for cluster_id in np.unique(clusters):                             # 존재하는 군집 라벨(0,1,2) 순회
    cluster_data=data[data['cluster']==cluster_id]                 # 해당 군집의 행만 필터링
    plt.scatter(cluster_data['annual_spending'],cluster_data['monthly_visits'],label=f'cluster{cluster_id}')    # x축: 연간 지출액, y축: 월 방문수, 범례 라벨


# 학습된 군집 중심(centroid) 표시
# - s=200 : 점 크기
# - c='black' : 색상
# - marker='X' : X 마커 모양
plt.scatter(Kmeans.cluster_centers_[:,0],                        # 각 중심의 x좌표
            Kmeans.cluster_centers_[:,1],                        # 각 중심의 y좌표
            s=200,c='black',marker='X',label='Centroids')
plt.xlabel('annual_spending')
plt.ylabel('monthly_visits')
plt.legend()
plt.show()