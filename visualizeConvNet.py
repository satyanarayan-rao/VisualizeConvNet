from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.datasets import mnist 

from ShapeHotEncoding import ascii_to_hd5_HotEncode

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
import scipy.misc

# number of channels
img_channels = 1

#%%
#  data
batch_size = 128 
nb_classes = 2
nb_epoch = 10 

nb_filters = 8 
nb_pool = 2
#nb_conv = 2
nb_conv_row = 4
nb_conv_col = 4
#%% Converting DNA letters to one-hot encoding matrix, and later converting them to images (black and white). 
Seq_HotEncode, Boolean_Response = ascii_to_hd5_HotEncode( filename= 'GSM1586782_ScrWT_Exd_16mer.txt',
                                                            add_shape= False,
                                                            add_reverse= False,
                                                            num_padding=0)

filename = 'GSM1586782_ScrWT_Exd_16mer.txt'
fp = open (filename)
image_file_names = [l.strip().split(' ')[0] for l  in fp] 
fp.close()
Seq_HotEncode, Boolean_Response, image_file_names = shuffle(Seq_HotEncode, 
                                                            Boolean_Response, 
                                                            image_file_names, 
                                                            random_state = 2)
#scipy.misc.imsave('outfile.jpg', Seq_HotEncode[1,:])
Seq_HotEncode_subset = Seq_HotEncode[0:20000,:]
Boolean_Response_subset = Boolean_Response[0:20000]
image_file_names_subset = image_file_names[0:20000]

#%%
#for i in range (0, Seq_HotEncode_subset.shape[0]):
#    image_name = 'images'+ '/' + image_file_names_subset[i] + '.png'
#    #scipy.misc.imsave(image_name, Seq_HotEncode_subset[i,:])
#    print (image_name)
#    scipy.misc.imsave(image_name, Seq_HotEncode_subset[i,:])

#%% Trying to see if we anyway need to convert into images
# original array
Seq_HotEncode_subset = Seq_HotEncode_subset.reshape (20000, 1, 4, 16) 
arr1 = Seq_HotEncode_subset[0,:] 
name = image_file_names_subset[0]
arr1 = arr1.reshape (1, 4, 16)

plt.imshow(arr1[0,:], interpolation = 'nearest')
print ("label: ", name)

# yes we can just use the array as it is in the case of sequence. No need to convert it to image file


#%% Now again reading images into matrix

#imlist = os.listdir('images')
#
#im1 = array(Image.open('images' + '/'+ imlist[0]))
#m,n = im1.shape[0:2]
#
#immatrix = array([array(Image.open('images'+ '/' + im2)).flatten()
#              for im2 in imlist],'f')
                  
train_data = [Seq_HotEncode_subset, Boolean_Response_subset]

(X, y) = (train_data[0],train_data[1])
y = [1 if x == True else 0 for x in y]
X_train, X_test, y_train, y_test, label_train, label_test = train_test_split(X,
                                                                             y,
                                                                             image_file_names_subset, 
                                                                             test_size=0.2,
                                                                             random_state=4)

img_rows, img_cols = X_train [0,0].shape
nb_classes = 2
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical (y_train, nb_classes)
Y_test = np_utils.to_categorical (y_test, nb_classes)

i = 200 
plt.imshow(X_train[i,0], interpolation = 'nearest')
print ("label: ", Y_train[i,:])
print ("image name: ", label_train[i])


#%% If you want to use MNIST data, you can uncomment lines in this block. 

#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train[0:10000,:]
#X_test  = X_test [0:1000,:]
#y_train = y_train [0:10000]
#y_test = y_test [0:1000]
#
#X_train = X_train.reshape (X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape (X_test.shape[0], 1, img_rows, img_cols)
#
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#
#X_train /= 255
#X_test /= 255
#
#print ('X_train shape: ', X_train.shape)
#
#print (X_train.shape[0], 'train samples')
#print (X_test.shape[0], 'test samples')
#
#Y_train = np_utils.to_categorical (y_train, nb_classes)
#Y_test = np_utils.to_categorical (y_test, nb_classes)
#
#i = 4600 
#plt.imshow (X_train[i, 0], interpolation = 'nearest')
#print ('label: ', Y_train[i,:])

#%%

model = Sequential()

model.add(Convolution2D(nb_filters, 4, 4,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))

convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, 4, 4))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
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


output_layer = model.layers[4].get_output()
output_fun = theano.function([model.layers[0].get_input()], output_layer)

input_image = X_train[0:1,:,:,:]
print (input_image.shape)
plt.imshow(input_image[0,0,:,:], interpolation='nearest', cmap='gray')

output_image = output_fun(input_image)
print (output_image.shape)

output_image =  np.rollaxis(np.rollaxis(output_image,3, 1 ), 3, 1)

fig = plt.figure(figsize = (8,8))
for i in range(8):
    ax = fig.add_subplot (3, 3, i+1)
    ax.imshow (output_image[0,:,:,i], cmap = matplotlib.cm.gray, interpolation='nearest')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()

plt     

