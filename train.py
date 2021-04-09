import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization

warnings.filterwarnings('ignore')

df = pd.read_csv('fer2013.csv')

dataset = df.values
X = df['pixels'].tolist()
XX = []
for xseq in X:
  tmp = [int(xp) for xp in xseq.split(' ')]
  tmp = np.asarray(tmp).reshape(48,48)
  XX.append(tmp.astype('float32'))
Y = pd.get_dummies(df['emotion']).values
# print (XX)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(XX, Y, test_size=0.2)
sample_shape = X_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

X_train = np.array(X_train).reshape(len(X_train),input_shape[0], input_shape[1], input_shape[2])
X_test = np.array(X_test).reshape(len(X_test),input_shape[0], input_shape[1], input_shape[2])

#Y_train = np.array(Y_train).astype('float32').reshape(-1,1)
#Y_test = np.array(Y_test).astype('float32').reshape(-1,1)

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Convolution2D(32, 4, 1, input_shape = (48, 48, 1), activation = 'relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 4, 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (4,2), strides=(2,2), padding='same'))
model.add(Dropout(0.1))

model.add(Convolution2D(32, 4, 1, activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 4, 1, activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (4,2), strides = (2,2), padding='same'))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 4, 1, activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 4, 1, activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (4,2), strides = (2,2), padding='same'))
model.add(Dropout(0.1))

model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(7, activation = 'sigmoid'))

#model.summary()

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

hist = model.fit(np.array(X_train), np.array(Y_train),
                 batch_size=64, 
                 epochs = 50,
                 validation_data = (np.array(X_test), np.array(Y_test)))

model.evaluate(X_test, Y_test)

y = Y_test
yhat = model.predict(X_test)
yh = yhat.tolist()
yt = y.tolist()
count = 0
predy=[]
truey=[]
for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(truey, predy)
print(cm)

model.save("model.h5")
