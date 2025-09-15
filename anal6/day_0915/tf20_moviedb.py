# 영화 리뷰 분류
# imdb dataset
from tensorflow.keras.datasets import imdb
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)
# !ls - al/root/.keras/datasets
print(train_data[:5])
print(train_label[:5])
print(train_data[0], len(train_data[0]))

# 참고 : 리뷰 데이터 하나를 원래 영어 단어로 보기
word_index = imdb.get_word_index()
print(word_index)
print(word_index.items())
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)
# print(reverse_word_index.get(train_data[0][:0])) 이건 빼는게 좋겟대 쌤이
decord_review = ' '.join([reverse_word_index.get(i) for i in train_data[1]])
print(decord_review)

# 데이터 준비
import numpy as np
def vector_seq(sequences, dim = 10000):
    results = np.zeros((len(sequences), dim))   # 크기가 (len(sequences), dim)이고 모든 값이 0인 행렬
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

x_train = vector_seq(train_data)    # train_data (list 타입을 vector(정확하게는 matrix)로 변환해준다)
x_test = vector_seq(test_data)
print(x_train, ' ', x_train.shape)  # (25000, 10000)

y_train = train_label.astype('float32')
y_test = test_label.astype('float32')

# 모델 작성
from tensorflow.keras import models, layers, regularizers

model = models.Sequential()
model.add(layers.Input(shape = (10000,)))
model.add(layers.Dense(units = 32, activation = 'relu'))
model.add(layers.Dense(units = 16, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
print(model.summary())