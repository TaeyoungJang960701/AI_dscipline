import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 학새 10명의 시험점수로 KMeans 수행 (K=3)
students=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
# 시험 점수
scores=np.array([76,95,65,85,60,92,55,88,83,72]).reshape(-1,1)
print('점수 : ',scores.ravel())

# KMeans 모델 생성
# - n_clusters=3 : 3개의 군집으로 나누기
# - random_state : 초기값 고정(재현성)
kmeans=KMeans(n_clusters=3,random_state=0)
kmeans_clust=kmeans.fit_predict(scores)

# 결과를 표(DataFrame)로 정리
# - Student: 학생 ID
# - Score  : 점수
# - Cluster: KMeans가 할당한 군집 라벨
df=pd.DataFrame({
    'Student':students,
    'Score':scores.ravel(),
    'Cluster':kmeans_clust
})
print('군집 결과 : \n',df)

print('군집별 평균 점수')
# 'Cluster' 값을 기준으로 행들을 묶고 → 각 묶음에서 'Score'의 평균을 계산
# groupby('Cluster') : 클러스터 라벨별로 그룹화
# ['Score']          : 각 그룹에서 'Score' 열만 선택
# mean()             : 선택한 열의 평균을 계산
grouped=df.groupby('Cluster')['Score'].mean()
print(grouped)

# 시각화
x_positions=np.arange(len(students))  # 학생 수만큼 x좌표 인덱스(0,1,2,...) 생성
y_scores=scores.ravel()               # 점수 배열을 1차원으로 펼쳐 사용
colors={0:'red',1:'blue',2:'black'}   # 군집 라벨 → 색상 매핑 딕셔너리

plt.figure(figsize=(10,6))
# - zip(x_positions, y_scores, kmeans_clust) : x/y좌표와 군집라벨을 묶어서 순회
for i,(x,y,cluster) in enumerate(zip(x_positions,y_scores,kmeans_clust)):
    plt.scatter(x,y,color=colors[cluster],s=100)
    # 산점도: 군집 라벨에 따라 색상을 선택하여 점 그리기
    plt.text(x,y+1.5,students[i],fontsize=10,ha='center')
    # 점 위에 학생 이름 표시
    # - y+1.5 : 점수값보다 살짝 위에 텍스트를 배치
    # - ha='center' : 수평 중앙 정렬
# plt.show()

# 중심점 표시
centers=kmeans.cluster_centers_
for center in centers:
    plt.scatter(len(students)//2,center[0],marker='X',c='gold',s=200)

plt.xticks(x_positions,students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.title('KMeans Clustering of students Scores')
plt.grid()
plt.show()