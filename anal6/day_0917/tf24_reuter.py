from keras.datasets import reuters
print(reuters.load_data(num_words = 10000))

(train_data, train_label), (test_data, test_label) = reuters.load_data(num_words = 10000)
print(len(train_data), len(test_data))      # 8982 2246
print(train_data[0])
print(train_label[0])
print(set(train_label))

# 실제 데이터 읽기
word_index = reuters.get_word_index()   # {'the' : 1}
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)

decord_review = ' '.join([reverse_word_index.get(i) for i in train_data[0]])
print(decord_review)

import numpy as np

def vector_seq(sequences, dim = 10000):
    results = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

x_train = vector_seq(train_data)
x_test = vector_seq(test_data)

print(x_test)

import sys

# np.set_printoptions(threshold = sys.maxsize)
print(x_test)

# one-hot encoding
"""
def to_onehot(labels, dim = 46):
    results = np.zeros((len(labels), dim))
    for i, lab in enumerate(labels):
        results[i, lab] = 1
    return results

one_hot_train_labels = to_onehot(train_label)
one_hot_test_labels = to_onehot(test_label)
print(one_hot_test_labels[0])
"""

from tensorflow.keras.utils import to_categorical
one_hot_train_labels = to_categorical(train_label)
one_hot_test_labels = to_categorical(test_label)
print(one_hot_test_labels[0])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import models

model = models.Sequential()
model.add(Input(shape = (10000,)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
print(model.summary())

# validation data
x_val = x_train[:1000]
partial_xtrain = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_ytrain = one_hot_train_labels[1000:]


history = model.fit(partial_xtrain, partial_ytrain,
                    epochs = 50, batch_size = 128, validation_data = (x_val, y_val), verbose = 2)

results = model.evaluate(x_test, one_hot_test_labels)
print(results)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']  # Corrected variable access
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'train loss')
plt.plot(epochs, val_loss, 'r', label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc'] # Corrected variable access
val_acc = history.history['val_acc'] # Corrected variable access

plt.plot(epochs, acc, 'bo', label = 'train acc')
plt.plot(epochs, val_acc, 'r', label = 'validation acc') # Corrected variable access
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()
plt.close()

