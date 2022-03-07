import pandas as pd
import matplotlib.pyplot as plt #visulization
from PIL import Image
from keras import models
import joblib
import numpy as np
import cv2
classifier=models.load_model("../CNN/custommodel.p")
#png_img = plt.imread("v2-00353.png",0)
img = cv2.imread("../CNN/v2-00440.png")
img = cv2.resize(img, (32, 32))
images = np.array(img)
def tienxuly(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img=cv2.equalizeHist(img)
    print("########################################")
    print(img)
    img=img/255
    return img
k=tienxuly(images)
k=k.reshape(1,32,32,1)
print(k.shape)
guess = classifier.predict_classes(k)
#print(guess)