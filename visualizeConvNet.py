from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.datasets import mnist 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

#%%
#  data
batch_size = 128 
nb_classes = 10 
nb_epoch = 10 
img_rows, img_cols = 28, 28 
nb_filters = 32 
nb_pool = 2 
nb_conv = 3 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[0:10000,:]
X_test  = X_test [0:1000,:]
y_train = y_train [0:10000]
y_test = y_test [0:1000]

X_train = X_train.reshape (X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape (X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print ('X_train shape: ', X_train.shape)

print (X_train.shape[0], 'train samples')
print (X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical (y_train, nb_classes)
Y_test = np_utils.to_categorical (y_test, nb_classes)

i = 4600 
plt.imshow (X_train[i, 0], interpolation = 'nearest')
print ('label: ', Y_train[i,:])

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))

convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(  X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            show_accuracy=True, verbose=1, validation_split=0.2)


score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])


def plot_filter (layer, x, y):
    filters = layer.W.get_value () 
    fig = plt.figure() 
    for j in range(len(filters)):
        ax = fig.add_subplot (y,x, j+1)
        ax.matshow (filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    return plt


output_layer = model.layers[0].get_output()
output_fun = theano.function([model.layers[0].get_input()], output_layer)

input_image = X_train[0:1,:,:,:]
print (input_image.shape)
plt.imshow(input_image[0,0,:,:], cmap = 'gray')

output_image = output_fun(input_image)
print (output_image.shape)

output_image =  np.rollaxis(np.rollaxis(output_image,3, 1 ), 3, 1)

fig = plt.figure(figsize = (8,8))
for i in range(32):
    ax = fig.add_subplot (6, 6, i+1)
    ax.imshow (output_image[0,:,:,i], cmap = matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()

plt     

