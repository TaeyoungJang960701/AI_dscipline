# 클러스터링 기법 중 계층적 군집화 이해
# 10명의 학생의 시험점수를 사용

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun gothic')
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
# linkage,     # 데이터로부터 링크지(linkage) 행렬을 계산.
#              # 군집 간 거리/결합 방식(method='ward','complete','average','single' 등)과
#              # 거리(metric='euclidean' 등)를 지정해 계층 구조를 만듦.

# dendrogram,  # linkage 행렬을 덴드로그램(나무 형태 그래프)으로 시각화.
#              # 군집 결합 순서·거리·임계선(cut line) 등을 한 눈에 파악 가능.

# fcluster     # linkage 행렬을 기준으로 '평탄한' 군집 라벨을 생성.
#              # 예) fcluster(Z, t=3, criterion='maxclust') → 3개 군집 라벨 반환
#              #     fcluster(Z, t=임계거리, criterion='distance') → 거리 기준으로 컷팅
students=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
# 시험 점수
scores=np.array([76,95,65,85,60,92,55,88,83,72]).reshape(-1,1)
print('점수 : ',scores.ravel()) # ravel() : 한줄로 표시 // 속도가 빠르다.

# 계층적 군집
linked=linkage(scores,method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked,labels=students)
plt.axhline(y=25,color='red',linestyle='--',label='cut at height=25')
plt.xlabel('students')
plt.ylabel('distance')
plt.legend()
plt.grid(True)
# plt.show()

# 군집 3개로 나누기
clusters=fcluster(linked,3,criterion='maxclust')
print(clusters)     # [2 1 3 1 3 1 3 1 1 2]
for student,cluster in zip(students,clusters):
    print(f'{student}:cluster {cluster}')

# 군집별로 점수와 이름 정리
cluster_info={}

for student, cluster, score in zip(students, clusters, scores.ravel()):     # students, clusters, scores를 동시에 순회 // scores.ravel(): 점수 배열을 1차원으로 평탄화(행렬/벡터 어떤 모양이든 안전하게 순회)
    if cluster not in cluster_info:
        cluster_info[cluster] = {'students': [], 'scores': []}      # 아직 해당 군집ID가 없으면 초기 구조 생성
    cluster_info[cluster]['students'].append(student)               # 현재 학생과 점수를 해당 군집에 누적
    cluster_info[cluster]['scores'].append(score)                   # 현재 학생과 점수를 해당 군집에 누적

for cluster_id, info in sorted(cluster_info.items()):       # 군집ID 순서대로(오름차순) 결과 요약 출력
    avg_score = np.mean(info['scores'])                     # 해당 군집의 평균 점수 계산
    student_list = ', '.join(info['students'])              # 학생 이름을 ", "로 연결한 문자열로 변환
    print(f'Cluster {cluster_id} : 평균점수={avg_score:.2f}, 학생들={student_list}')        # f-string으로 깔끔하게 출력 (평균점수는 소수 둘째자리까지)

# 군집 시각화
x_positions=np.arange(len(students))
y_Scores=scores.ravel()
colors={1:'red',2:'blue',3:'gray'}
plt.figure(figsize=(10,6))

for i,(x,y,cluster) in enumerate(zip(x_positions,y_Scores,clusters)):
    plt.scatter(x,y,color=colors[cluster],s=100)
    plt.text(x,y+1.5,students[i],fontsize=10,ha='center')
plt.xticks(x_positions,students)
plt.xlabel('students')
plt.ylabel('score')
plt.grid()
plt.show()