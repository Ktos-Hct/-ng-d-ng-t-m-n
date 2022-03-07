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
path='data5'
chisodaotao=0.2
imageDimensions=(32,32,3)
##
images=[]
classNo=[]
mylist=os.listdir(path)
sothumuc=len(mylist)
print("Tong so thu muc " ,len(mylist))
for i in range(0,sothumuc):
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

somau=[]
for i in range(0,sothumuc):
    #print(len(np.where(y_train==i)[0]))
    somau.append(len(np.where(y_train==i)[0]))
print(somau)
print(len(somau))
plt.figure(figsize=(10,5))
plt.bar(mylist,somau)
plt.title("thong ke so luong mau")
plt.xlabel("triso")
plt.ylabel("so hinh anh")
plt.show()

def tienxuly(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img
#img=tienxuly(x_train[100])

x_train=np.array(list(map(tienxuly,x_train)))
x_test=np.array(list(map(tienxuly,x_test)))
x_validation=np.array(list(map(tienxuly,x_validation)))

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)
#
datagen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
datagen.fit(x_train)
print(classNo)
y_train=to_categorical(y_train,sothumuc)
y_test=to_categorical(y_test,sothumuc)
y_validation=to_categorical(y_validation,sothumuc)
#

def model_2():
    soboloc=60
    boloc1=(5,5)
    boloc2=(3,3)
    tong=(2,2)
    sonut=500
    model=Sequential()
    model.add((Conv2D(soboloc,boloc1,input_shape=(imageDimensions[0],
                                                  imageDimensions[1],
                                                  1),activation='relu'
                                                    )))
    model.add((Conv2D(soboloc,boloc1,activation='relu')))
    model.add(MaxPooling2D(pool_size=tong))
    model.add((Conv2D(soboloc // 2,boloc2,activation='relu')))
    model.add((Conv2D(soboloc // 2,boloc2,activation='relu')))
    model.add(MaxPooling2D(pool_size=tong))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(sonut,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(sothumuc,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model
model=model_2()

print(model.summary())
bathSizeVal=50
epochsval=1
step=2000
history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=bathSizeVal),
                                    steps_per_epoch=step,
                                    epochs=epochsval,
                                    validation_data=(x_validation,y_validation),
                                    shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()
score=model.evaluate(x_test,y_test,verbose=0)
print('Test Score= ',score[0])
print('test accuracy= ',score[1])
model.save('custommodel113.p')
#print('testscore',score[0])
#print('testscore',score[1])

#joblib.dump(model,"model2.p")
#print(y_test)
