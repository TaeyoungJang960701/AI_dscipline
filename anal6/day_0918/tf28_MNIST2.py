# MNIST로 학습된 모델로 내가 그린 숫자 이미지 분류 확인
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# im = Image.open('su.png')
im = Image.open('/content/sample_data/su.png')
# 원래 이미지 크기 28 * 28 크기로 리사이즈(MNIST 기준)
# 흑백(0 ~ 255)으로 변환 후 numpy 배열로 변환

img = np.array(im.resize((28, 28), Image.Resampling.LANCZOS).convert('L'))
print(img.shape)

plt.imshow(img, cmap = 'Greys')
plt.show()
# 아래 그림판을 채널 이라고 부른대 아래는 검은색밖에 없으니까 채널이 하나
# RGB값 (x, x, x) 이건 채널이 세개인거지

# (28 * 28) 이미지를 (1, 784) 벡터로 변환 (Dense 클래스 입력 형태)
data = img.reshape([1, 784]).astype('float32')
print(data)
print('-' * 100)

data = data / 255.0     # 픽셀값을 0에서 더미로 정규화시킴
print(data)

# 다시 시각화 (1, 784) -> (28, 28)      reshape하면 된다
# 숫자 쪼가리로 됏던 이미지를 다시 그림으로
plt.imshow(data.reshape(28, 28), cmap = 'Greys')
plt.show()

# 저장된 숫자 이미지 분류 모델 읽어 내 이미지 분류해보기
import tensorflow as tf

save_path = '/content/drive/MyDrive/mysou/tf27_model.keras'
mymodel = tf.keras.models.load_model(save_path)

new_pred = mymodel.predict(data, verbose = 0)
print('new_pred : ', new_pred)
print('new_pred : ', np.argmax(new_pred,1))
