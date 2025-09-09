# iris dataset으로 지도학습(K_NN)/비지도 학습(K-Means) 비교

from sklearn.datasets import load_iris

# 아이리스(붓꽃) 데이터셋 로드: dict-유사 Bunch 객체 반환
iris_dataset=load_iris()

# 사용 가능한 키 확인: ['data','target','feature_names','target_names','DESCR', ...]
print(iris_dataset.keys())

print(iris_dataset['data'][:3])     # 특징 행렬 X의 앞 3행 미리 보기 (샘플 수 × 특징 수=4)
print(iris_dataset['feature_names'])        # 특징 이름(4개): sepal length/width, petal length/width
print(iris_dataset['target'][:3])   # 정답 라벨 y의 앞 3개 (0,1,2는 품종 인덱스)
print(iris_dataset['target_names'])         # 라벨 이름: ['setosa' 'versicolor' 'virginica']


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(iris_dataset['data'],iris_dataset['target'],test_size=0.25,random_state=42)  # 학습/테스트 데이터 분할 (테스트 25%, 난수 고정=42)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)        # (112, 4) (38, 4) (112,) (38,)

print('지도학습 : K-NN ------------------ ')
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics

knnModel=KNeighborsClassifier(n_neighbors=3, weights='distance',metric='euclidean')
# K-최근접이웃 분류기 설정
# - n_neighbors=3 : 이웃 3개 사용
# - weights='distance' : 가까운 이웃에 더 큰 가중치(거리 가중)
# - metric='euclidean' : 유클리드 거리(기본값 minkowski(p=2)와 동일)
# distance -> euclidean // unform -> minkowski
knnModel.fit(train_x,train_y)   # feature, label(tag,target,class)로 학습

predict_label=knnModel.predict(test_x)
print('예측값 : ',predict_label)        # [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1 0]
print('test acc : {:.3f}'.format(np.mean(predict_label==test_y)))       # test acc : 1.000 // 간단 정확도 계산 (방법 1: 평균)
print('acc : ',metrics.accuracy_score(test_y,predict_label))            # acc :  1.0 // 간단 정확도 계산 (방법 2: metrics 함수)

# 새로운 데이터 분류
new_input=np.array([[6.1,2.8,4.7,1.2]])     # 반드시 2차원 형태(샘플 수 × 특징 수)
print(knnModel.predict(new_input))          # 예측 라벨 인덱스(예: [1])
print(knnModel.predict_proba(new_input))    # 각 클래스(0/1/2)에 대한 확률 분포
dist,index=knnModel.kneighbors(new_input)   
print(dist,index)       # [[0.2236068  0.3        0.43588989]] [[71 82 31]]

print('비지도학습 : KMeans(데이터에 정답(label)이 없는 경우) ------------------ ')
from sklearn.cluster import KMeans
# KMeans 군집화 모델(클러스터 개수=3)
# - init='k-means++' : 안정적인 초기 중심 선택 방법
# - n_init=10        : 서로 다른 초기값으로 10번 실행 후 가장 좋은 해 선택
# - random_state=0   : 재현성(난수 고정)
kmeansModel=KMeans(n_clusters=3,init='k-means++',n_init=10,random_state=0)
kmeansModel.fit(train_x)    # 비지도 학습: 정답(train_y) 없이 특징(train_x)만으로 군집 학습
# print(kmeansModel.labels_)
print('0번째 cluster:',train_y[kmeansModel.labels_==0])
print('1번째 cluster:',train_y[kmeansModel.labels_==1])
print('2번째 cluster:',train_y[kmeansModel.labels_==2])

# 이번엔 클러스팅에서 새로운 데이터 분류
new_input=np.array([[6.1,2.8,4.7,1.2]])
clu_pred=kmeansModel.predict(new_input)
print(clu_pred)     # [2]

