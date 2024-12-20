import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

################# Parameters #####################
path = "myData" # folder with all the class folders
labelFile = 'labels.csv' # file with all names of classes
batch_size_val=50  # how many to process together
steps_per_epoch_val=2000
epochs_val=10
imageDimesions = (32,32,3)
testRatio = 0.2    # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
###################################################

############################### Importing of the Images
count = 0
images = []
classNo = []
myList = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(len(myList)):
    try:
        myPicList = os.listdir(os.path.join(path, str(count)))
        for y in myPicList:
            curImg = cv2.imread(os.path.join(path, str(count), y))
            images.append(curImg)
            classNo.append(count)
        print(count, end=" ")
    except FileNotFoundError:
        print(f"Skipping missing folder: {count}")
    count += 1

print(" ")
images = np.array(images)
classNo = np.array(classNo)

############################### Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data Shapes")
print("Train",end = ""); print(X_train.shape, y_train.shape)
print("Validation",end = ""); print(X_validation.shape, y_validation.shape)
print("Test",end = ""); print(X_test.shape, y_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "The number of images in not equal to the number of labels in training set"
assert(X_validation.shape[0] == y_validation.shape[0]), "The number of images in not equal to the number of labels in validation set"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of labels in test set"
assert(X_train.shape[1:] == (imageDimesions)), " The dimensions of the Training images are wrong "
assert(X_validation.shape[1:] == (imageDimesions)), " The dimensions of the Validation images are wrong "
assert(X_test.shape[1:] == (imageDimesions)), " The dimensions of the Test images are wrong"

############################### PREPROCESSING THE IMAGES

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))  # TO iterate and preprocess all images
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

############################### ADD A DEPTH OF 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

############################### ONE-HOT ENCODING OF LABELS
y_train = to_categorical(y_train, num_classes=noOfClasses)
y_validation = to_categorical(y_validation, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)

############################### CONVOLUTION NEURAL NETWORK MODEL
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu'))
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

############################### TRAIN
model = myModel()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=epochs_val, batch_size=batch_size_val)

############################### Save the Model
# Save the model using Keras' save method (recommended method)
model.save("model_trained.h5")

# Save the model as a pickle object (optional, but not recommended for Keras models)
with open("model_trained.p", "wb") as pickle_out:
    pickle.dump(model, pickle_out)

print("Model saved successfully.")
