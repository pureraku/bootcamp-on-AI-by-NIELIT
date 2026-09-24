import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg")
print(type(img))
print(img.shape)

cv2.imshow("pho",img[100:300,200:500])
cv2.waitKey(0)

img_resize = cv2.resize(img, (256,256))
img_flip = cv2.flip(img_resize, 1)
cv2.imshow("photo",img_flip)
cv2.waitKey(0)

