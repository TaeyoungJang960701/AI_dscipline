# 세계 정치인들 얼굴 사진을 이용한 분류 모델
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

faces = fetch_lfw_people(min_faces_per_person=60, color = False, resize = 0.5)
# print(faces.DESCR)

print(faces.data)
print(faces.data.shape)     # (1348, 2914)
print(faces.target)         
print(faces.target_names)   # ['Ariel Sharon' 'Colin Powell' . . .
print(faces.images.shape)   # (1348, 62, 47) 62x47 사이즈 사진이 1348장 있다

print(faces.images[567])
print(faces.target_names[faces.target[567]])
# plt.imshow(faces.images[567], cmap = 'bone')
# plt.show()
# plt.close()

"""
fig, ax = plt.subplots(3,5)
# print(fig)
# print(ax.flat)
# print(len(ax.flat))   # 15
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap = 'bone')
    axi.set(xticks = [], yticks = [], xlabel = faces.target_names[faces.target[i]])
plt.show()
"""

# 주성분분석으로 이미지 차원을 축소시켜 분류 작업 진행
m_pca = PCA(n_components=150, whiten = True, random_state=0)
x_low = m_pca.fit_transform(faces.data)
print('x_low : ', x_low, ' ', x_low.shape)      # (1348, 150) 2914 -> 150

m_svc = SVC(C = 1)
model = make_pipeline(m_pca, m_svc)
print(model)
# Pipeline(steps=[('pca', PCA(n_components=150, random_state=0, whiten=True)),
#                 ('svc', SVC(C=1))])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
    train_test_split(faces.data, faces.target, test_size=0.3, random_state=1)

model.fit(x_train, y_train)
pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10])

print('-' * 100)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, target_names = faces.target_names))

from sklearn.metrics import confusion_matrix, accuracy_score
mat = confusion_matrix(y_test,pred)
print('confusion_matrix : \n', mat)
print('acc : \n', accuracy_score(y_test,pred))  #  0.7185185185185186
print('-' * 100)

# # 분류 결과 시각화
# # x_test[0] 한개만 확인 
# print(x_test[0], ' ', x_test[0].shape)
# print(x_test[0].reshape(62,47))     # 이미지 출력시 이처럼 1차원을 2차원으로 변환
# plt.subplots(1, 1)
# plt.imshow(x_test[0].reshape(62,47), cmap = 'bone')
# plt.show()
# plt.close()
fig, ax = plt.subplots(4,6)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62,47), cmap = 'bone')
    axi.set(xticks = [], yticks = [])
    axi.set_ylabel(faces.target_names[pred[i]].split()[-1],\
                   color = 'black' if pred[i] == y_test[i] else 'red')
    fig.suptitle('pred result', size = 14)
plt.show()

# 오차 행렬 시각화
import seaborn as sns
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False,\
            xticklabels = faces.target_names, yticklabels = faces.target_names)
plt.xlabel('true(real) label')
plt.xlabel('pred label')
plt.show()
plt.close()