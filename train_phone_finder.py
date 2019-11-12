#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Aravind Manoharan
aravindmanoharan05@gmail.com
https://github.com/aravindmanoharan
'''

import sys
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

IMAGE_FOLDER = sys.argv[1]
DATA_FILE = IMAGE_FOLDER + "/labels.txt"

def load_data(DATA_FILE):
    f = open(DATA_FILE, 'r')
    info = []
    for line in f:
        info.append(line.split())
        
    data = pd.DataFrame(info, columns=['image_name', 'x', 'y'])
    data[['x','y']] = data[['x','y']].astype('float')
    
    images = []
    labels = []
    image_h = 0
    image_w = 0
    
    for index, row in data.iterrows():
        image = cv2.imread(IMAGE_FOLDER + '/' + row['image_name'])
        
        if image_h == 0 or image_w == 0:
            image_h = image.shape[0]
            image_w = image.shape[1]
            
        images.append(image)
        labels.append([int(row['x']*image_w), int(row['y']*image_h)])
    
    return images, labels

def extract_windows(images, labels, stride = 5):

    image_h = images[0].shape[0]
    image_w = images[0].shape[1]
    
    windows = []
    windows_labels = []
    
    for n in range(len(images)):
        for y in range(0, image_h, stride):
            for x in range(0, image_w, stride):
                if (x + 40) < image_w and (y + 40) < image_h:
                    top_left = (x, y)
                    bottom_right = ((x + 40), (y + 40))
                    
                    if (labels[n][0] >= top_left[0] and labels[n][0] <= bottom_right[0]) and \
                        (labels[n][1] >= top_left[1] and labels[n][1] <= bottom_right[1]):
                        windows_labels.append(1)
                    else:
                        windows_labels.append(0)
                    
                    img = images[n][y:y+40,x:x+40]
                    windows.append(img)
    
    return np.array(windows), np.array(windows_labels)

def removing_non_phone_images(windows_labels, n_sample = 653380):
    
    removable_rows = []
    n_sample = 653380
    
    for win_label in range(len(windows_labels)):
        if windows_labels[win_label] == 0:
            removable_rows.append(win_label)
    
    removing_rows = random.sample(removable_rows, n_sample)
    
    return removing_rows

def data_split(images, labels, test_ratio = 0.2):
    
    # Shuffle the train images and split into trian and test data
    images, labels = shuffle(images, labels)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, random_state=0, test_size=test_ratio)

    # Reshape the labels too
    train_labels = train_labels.reshape(train_labels.shape[0], 1)
    test_labels = test_labels.reshape(test_labels.shape[0], 1)
    
    return train_images, test_images, train_labels, test_labels

def model_architecture(image_x, image_y):
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(20,(5,512), padding='same', input_shape=(image_x, image_y, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution layer
    model.add(Conv2D(50,(5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Fully connected layer 2nd layer
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    filepath = "model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    ES = EarlyStopping(monitor='val_loss', patience=5)
    callbacks_list = [ES, checkpoint]

    return model, callbacks_list

if __name__ == '__main__':
        
    images, labels = load_data(DATA_FILE)
    
    windows, windows_labels = extract_windows(images, labels, stride = 5)
    
    print("Windows shape: {}".format(windows.shape))
    print("Labels shape: {}".format(windows_labels.shape))
    
    training_size = 20000
    removing_rows = removing_non_phone_images(windows_labels, n_sample = len(windows) - training_size)
    
    windows = np.delete(windows, removing_rows, 0)
    windows_labels = np.delete(windows_labels, removing_rows, 0)
    
    print("Windows shape: " + str(windows.shape))
    print("Labels shape: " + str(windows_labels.shape))
    
    print("Ratio of positive labels: {}".format(int(np.sum(windows_labels) / len(windows_labels) * 100)))
    
    # Convert the images and labels into numpy array
    images = np.array(windows).astype('float32')
    images = images/np.max(images)
    labels = np.array(windows_labels).astype('float32')
    
    print("Images shape: " + str(images.shape))
    print("Labels shape: " + str(labels.shape))
    
    train_images, test_images, train_labels, test_labels = data_split(images, labels, test_ratio = 0.2)
    
    print("Number of Training examples: " + str(train_images.shape[0]))
    print("Number of Testing examples: " + str(test_images.shape[0]))
    
    print("train_images shape: " + str(train_images.shape))
    print("test_images shape: " + str(test_images.shape))
    
    print("train_labels shape: " + str(train_labels.shape))
    print("test_labels shape: " + str(test_labels.shape))
    
    model, callbacks_list = model_architecture(train_images.shape[1], train_images.shape[2])
    
    print(model.summary())
    
    epochs = 5
    training_model = model.fit(train_images, train_labels, validation_split=0.1, shuffle=True, epochs=epochs, \
                               batch_size=128, callbacks=callbacks_list, verbose=1)
    
    model.save_weights('final_weights.h5')
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("Model trained and saved")
      
    loss = training_model.history['loss']
    val_loss = training_model.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
