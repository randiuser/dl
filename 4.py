import cv2
import numpy as np
from skimage import exposure, filters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

img = cv2.imread('example.jpg', 0)
ops = {
    'Original': img,
    'Equalized': exposure.equalize_hist(img),
    'Thresholded': cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
    'Edges': filters.sobel(img),
    'Augmented': ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True).flow(img.reshape(1, *img.shape, 1), batch_size=1).next()[0].squeeze(),
    'Dilated': cv2.dilate(img, np.ones((5,5), np.uint8), iterations=1),
    'Eroded': cv2.erode(img, np.ones((5,5), np.uint8), iterations=1)
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for (title, operation), ax in zip(ops.items(), axes.ravel()):
    ax.imshow(operation, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
