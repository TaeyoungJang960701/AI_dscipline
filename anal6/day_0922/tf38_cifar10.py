# cifar-10 dataset으로 이미지 분류 모델을 작성(CNN 사용 X)
# 총 10개의 label과 6만장의 color 이미지 학습. 32 * 32
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print('샘플 수 : ',x_train.shape)         # 샘플 수 :  (50000, 32, 32, 3)
print('채널 수 : ',x_train.shape[3])      # 채널 수 :  3
print('이미지 크기 : ',x_train.shape[1],x_train.shape[2])   # 이미지 크기 :  32 32
print('test 샘플 수 : ',x_test.shape)     # test 샘플 수 :  (10000, 32, 32, 3)
print('test type : ',x_test.dtype)        # test type :  uint8

# print(x_train[0])
print(y_train[0])

# 시각화
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(x_train[0],interpolation='bicubic')
plt.subplot(132)
plt.imshow(x_train[1],interpolation='bicubic')
plt.subplot(133)
plt.imshow(x_train[2],interpolation='bicubic')
plt.show()

# feature 원-핫 인코딩
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
# print(x_train[0])

# label 원-핫 인코딩
NUM_CLASSES = 10
y_train = to_categorical(y_train,NUM_CLASSES)
y_test = to_categorical(y_test,NUM_CLASSES)
# print(y_train[0])

# 모델생성
"""
model=Sequential([
    Input(shape=(32,32,3)),
    Flattne(),
    Dense(units=256,activation='relu'),
    Dense(units=128,activation='relu'),
    Dense(units=NUM_CLASSES,activation='softmax')
])
print(model.summary())
"""

Input_layer=Input(shape=(32,32,3))
x=Flatten()(Input_layer)
x=Dense(units=256,activation='relu')(x)
x=Dense(units=128,activation='relu')(x)
output_layer=Dense(units=NUM_CLASSES,activation='softmax')(x)
model=Model(Input_layer,output_layer)
print(model.summary())

# train
opt=Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=20,shuffle=True,verbose=2)

print('test acc : %.4f'%(model.evaluate(x_test,y_test,verbose=0,batch_size=64)[1]))
print('test loss : %.4f'%(model.evaluate(x_test,y_test,verbose=0,batch_size=64)[0]))

CLASSES=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

# 예측
pred = model.predict(x_test[:10])
print(np.argmax(pred, axis = -1))

pred_cla = CLASSES[np.argmax(pred, axis = -1)]
actual_cla = CLASSES[np.argmax(y_test[:10], axis = -1)]

print('예측값 : ', pred_cla)
print('실제값 : ', actual_cla)
print('분류 실패 수 : ', (pred_cla != actual_cla).sum())

fig = plt.figure(figsize = (15, 3))
# fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

for i, idx in enumerate(range(len(x_test[:10]))):
    img = x_test[idx]
    ax = fig.add_subplot(1, len(x_test[:10]), i + 1)
    ax.axis('off')
    ax.text(0.5, -0.3, 'pred = ' + str((pred_cla[idx])), 
                                       fontsize = 10, ha = 'center', transform = ax.transAxes)
    ax.text(0.5, -0.6, 'act = ' + str((actual_cla[idx])), 
                                       fontsize = 10, ha = 'center', transform = ax.transAxes)
    ax.imshow(img)
plt.show()

