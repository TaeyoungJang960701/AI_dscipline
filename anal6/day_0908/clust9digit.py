# 숫자 이미지 데이터에 K-평균 알고리즘 사용하기
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()      # 64개의 특징(feature)을 가진 1797개의 표본으로 구성된 숫자 데이터
print(digits.data.shape)  # (1797, 64) 64개의 특징은 8*8 이미지의 픽셀당 밝기를 나타냄
 
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)  # (10, 64)  # 64차원의 군집 10개를 얻음


# 군집중심이 어떻게 보이는지 시각화
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest')
plt.show()  # 결과를 통해 KMeans가 레이블 없이도 1과 8을 제외하면 

# 인식 가능한 숫자를 중심으로 갖는 군집을 구할 수 있다는 사실을 알 수 있다. 
# k평균은 군집의 정체에 대해 모르기 때문에 0-9까지 레이블은 바뀔 수 있다.
# 이 문제는 각 학습된 군집 레이블을 그 군집 내에서 발견된 실제 레이블과 매칭해 보면 해결할 수 있다.

from scipy.stats import mode
labels = np.zeros_like(clusters)

for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# 정확도 확인
from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target, labels))  # 0.79354479

# 오차행렬로 시각화
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()  # 오차의 주요 지점은 1과 8에 있다.

# 참고로 t분포 확률 알고리즘을 사용하면 분류 정확도가 높아진다.
from sklearn.manifold import TSNE 

# 시간이 약간 걸림
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(digits.data) 

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)

for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# Compute the accuracy
print(accuracy_score(digits.target, labels))  # 0.93266555