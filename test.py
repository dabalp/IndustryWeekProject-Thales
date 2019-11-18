from sklearn import datasets
import cv2
import numpy as np


X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='data/')

img = np.array(X[0]).reshape((28, 28))
img = cv2.resize(img, (200, 200))
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()