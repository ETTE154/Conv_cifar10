
#%%

# -*- coding: utf-8 -*-
from tensorflow import keras
from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

#====================================================================================================
#%%
# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()

print(trainX.shape, trainY.shape, testX.shape, testY.shape)
#====================================================================================================
#%%
np.random.seed(123)


class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

sample_size = 9 # 3x3
random_idx = np.random.choice(trainX.shape[0], sample_size) # 0~49999

plt.figure(figsize=(5,5)) # 7x7
for i, idx in enumerate(random_idx): # 0~8
    plt.subplot(3,3,i+1)  # 3x3 i+1번째 그래프
    plt.imshow(trainX[idx], interpolation='nearest')    # interpolation='nearest' : 픽셀을 그대로 표현
    plt.title("{}".format(class_names[trainY[idx][0]]), fontsize=12)    # trainY[idx][0] : 0~9
    plt.axis('off')     # x, y 축 눈금 제거
plt.tight_layout()      # 그래프 간격 조정
plt.show()
#====================================================================================================
# %%
# 데이터셋 전처리 과정
# 평균과 표준편차를 이용한 정규화
x_mean = np.mean(trainX, axis=(0,1,2))
x_std = np.std(trainX, axis=(0,1,2))

x_train = (trainX - x_mean) / x_std
x_test = (testX - x_mean) / x_std


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x_train, trainY, test_size=0.3, random_state=123)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
#====================================================================================================
# %%

# 모델 구성
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3))) # 필터크기 3x3, 패딩, 활성화함수, 입력크기
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size= 2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())    # 1차원으로 변환, 1차원으로 변환한 후에는 Dense로 연결
model.add(Dense(256, activation='relu')) # 256개의 노드, 활성화함수 relu
model.add(Dense(10, activation='softmax')) # 10개의 클래스로 분류, 여러게의 클래스로 분류할 때는 softmax 사용

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam, loss, metrics
#====================================================================================================
# %%
# 성능향상이 멈추면 학습을 멈추는 조기종료
# 에폭 마다 가중치 저장
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

early_stopping = EarlyStopping(monitor = 'val_acc', patience=10) # 10번의 에폭동안 성능향상이 없으면 학습을 멈춤
modelpath = 'model/best_cifar10.h5' # 가중치 저장 경로
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True) # val_loss가 가장 좋은 가중치를 저장
#====================================================================================================

# 모델 학습
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpointer])
# %%

his_dict = history.history # 훈련 데이터의 손실값, 정확도, 검증 데이터의 손실값, 정확도
loss = his_dict['loss'] # 훈련 데이터의 손실값

val_loss = his_dict['val_loss'] # 검증 데이터의 손실값

epochs = range(1, len(loss)+1) # 1~30
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax1.plot(epochs, loss, 'orange', label='Training loss') # 오랜지색 실선
ax1.plot(epochs, val_loss, 'b', label='Validation loss') # 파란색 실선
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

acc = his_dict['accuracy'] # 훈련 데이터의 정확도
val_acc = his_dict['val_accuracy'] # 검증 데이터의 정확도

ax2 = fig.add_subplot(1,2,2)
ax2.plot(epochs, acc, 'orange', label='Training accuracy') # 오랜지색 실선
ax2.plot(epochs, val_acc, 'r', label='Validation accuracy') # 빨간색 실선
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()
#====================================================================================================
# %%
