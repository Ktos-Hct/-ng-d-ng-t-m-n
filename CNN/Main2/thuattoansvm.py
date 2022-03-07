import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm
import joblib
import cv2
data_to_train = []
values_of_data = []

training_set_size = 3200
hop_size = 30
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import joblib
path='data3'
chisodaotao=0.2
imageDimensions=(32,32,3)
##
images=[]
classNo=[]
mylist=os.listdir(path)
sothumuc=len(mylist)
print("Tong so thu muc " ,len(mylist))
for i in mylist:
    mypiclist=os.listdir(path+"/"+str(i))
    for j in mypiclist:
        img=cv2.imread(path+"/"+str(i)+"/"+j)
        print(img.shape)
        img=cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(i)
    print(i,end=" ")
print("ketthuc")
print(classNo)
images = np.array(images)
classNo=np.array(classNo)
print(classNo.shape)
x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=chisodaotao)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=chisodaotao)
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)
classifier = svm.SVC(gamma=0.001, C=100)
classifier.fit(data_to_train, values_of_data)
joblib.dump(classifier,"model3.pkl")
print ("Done")
