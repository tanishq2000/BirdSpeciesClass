# -*- coding: utf-8 -*-
"""Bird_species_project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NMHEQc9JaTGm-IF6pWXAv8mnTgo1QMLz
"""

import zipfile

with zipfile.ZipFile('Bird Speciees Dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()

import os
import cv2
import imghdr
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense

#from tensorflow.keras.applications import MobileNet
#from tensorflow.keras import layers
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from tensorflow.keras.models import Model
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import SGD

image_exts = ['jpeg','jpg', 'bmp', 'png']
def load_data(directory):
    images = [] #Here we are storing the actual images
    labels = [] #Here we store the labels or classes of those images
    imge_path = [] #Here we are also storing the image paths
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        print(class_path)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            print(img_path)
            #Checking if the image is having the compatible extension
            try:
                img = cv2.imread(img_path)
                tip = imghdr.what(img_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(img_path))
                    os.remove(img_path)
            except Exception as e:
                print('Issue with image {}'.format(img_path))
                continue
                #os.remove(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(class_name)
            imge_path.append(img_path)
    return images, labels, imge_path
            #img = cv2.imread(img_path)
            # Resize images if necessary

#Storing path to the data set into a variable
dataset_dir = 'Bird Speciees Dataset/'
image, labels, imge_path = load_data(dataset_dir)

# Converting the lists into numpy arrays
images = np.array(image)
labels = np.array(labels)

""" Here we use train_test_split function from matplotlib library to split our dataSet into training and testing dataSets
 in it we split our test_size to 20% and shuffel the dataset before split using random_state function ensuring that the data is split
 consistently  each time"""
x_train, x_test, y_train, y_test, train_img_path, test_img_path = train_test_split(images, labels, imge_path, test_size=0.2, random_state=42)

for i in range(5):
    plt.imshow(cv2.cvtColor(x_train[i], cv2.COLOR_BGR2RGB))
    plt.title(y_train[i])
    plt.axis('on')
    plt.show()

for i in range(5):
    plt.imshow(cv2.cvtColor(x_test[i], cv2.COLOR_BGR2RGB))
    plt.title(y_test[i])
    plt.axis('on')
    plt.show()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = x_train/255.0
x_test = x_test/255.0

x_train, x_test

# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

label_counts = pd.DataFrame(labels).value_counts()
label_counts
num_classes = len(label_counts)
num_classes

# Building model architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same",input_shape=(224,224,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0005),metrics=['accuracy'])

# Training the model
epochs = 50
batch_size = 128
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))

#Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(labels = ['train', 'val'], loc = "lower right")
plt.show()

#Plot the loss history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'], color='b')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

# Calculating test accuracy
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

# Storing predictions
y_pred = model.predict(x_test)
labels[np.argmax(y_pred[7])]

# Plotting image to compare
img = array_to_img(x_test[7])
img

# Finding max value from predition list and comaparing original value vs predicted
labels = label_binarizer.classes_
print(labels)
print("Originally : ",labels[np.argmax(y_test[7])])
print("Predicted : ",labels[np.argmax(y_pred[7])])

# Saving model
model.save("bird_species.h5")

