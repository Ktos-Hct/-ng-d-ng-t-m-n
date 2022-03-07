import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import models
cascade = cv2.CascadeClassifier("custom.xml")
img=cv2.imread('Main2/51.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bienso = cascade.detectMultiScale(gray, 1.1, 4)
cv2.imshow('dulieugoc',bienso)
cv2.waitKey(0)
cv2.destroyWindow()