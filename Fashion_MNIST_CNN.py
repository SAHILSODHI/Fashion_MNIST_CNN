#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:18:28 2019

@author: sahilsodhi
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

# Load the fashion-mnist train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape)

plt.imshow(x_train[0])

fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Image index, you can pick any number between 0 and 59,999
img_index = 10
# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]
# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
# # Show one of the images from the training dataset
plt.imshow(x_train[img_index])

#Normalize data dimensions so that the images are of same scale between 0 and 1.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_valid = to_categorical(y_valid, 10)
y_test = to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

#Initializing the CNN model.
CNNmodel = Sequential()

# Adding Convolution layer 1.
CNNmodel.add(Convolution2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
# Adding MaxPooling layer 2.
CNNmodel.add(MaxPooling2D(pool_size=2))
# Adding a dropout layer 3 - This layer “drops out” a random set of activations in that layer by setting them to zero.
CNNmodel.add(Dropout(0.3))
# Adding Convolution layer 4.
CNNmodel.add(Convolution2D(filters=32, kernel_size=2, padding='same', activation='relu'))
# Adding MaxPooling layer 5.
CNNmodel.add(MaxPooling2D(pool_size=2))
# Adding a dropout layer 6
CNNmodel.add(Dropout(0.3))
# Flattening... put all pooled feature maps to a vector. Input to ANN.
CNNmodel.add(Flatten())
# Adding hidden layer 7.
CNNmodel.add(Dense(256, activation='relu'))
# Adding a dropout layer 8.
CNNmodel.add(Dropout(0.5))
# Adding hidden layer 9.
CNNmodel.add(Dense(10, activation='softmax'))

# Take a look at the model summary
CNNmodel.summary()

# Compiling the CNN
CNNmodel.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Fitting the CNN to the images
result = CNNmodel.fit(x_train,
         y_train,
         batch_size=64,
         epochs=20,
         validation_data=(x_valid, y_valid))

# Evaluate the model on test set
score = CNNmodel.evaluate(x_test, y_test, verbose=0)

# Print test accuracy.
print('\n', 'Test accuracy:', score[1])

y_hat = CNNmodel.predict(x_test)

# Plotting accuracy using summarised history for accuracy
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy (Train and Test) vs Epochs')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.grid()
plt.show()

# Plotting loss using summarised history for accuracy
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss (Train and Test) vs Epochs')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.grid()
plt.show()
