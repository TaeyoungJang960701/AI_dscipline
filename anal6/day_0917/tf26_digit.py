# 손글씨(숫자 이미지) 읽기
# https://www.tensorflow.org/datasets/catalog/mnist?hl=ko
# su.png는 위 링크에서 가져온 손글씨 이미지 아무거나 가져온거야
# 그걸 코랩 왼쪽 창의 파일 칸에 드래그 앤 드롭 한거

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('su.png')
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

