import numpy as np
import cv2
from skimage import filters

# Generate image with 2 objects and background
image = np.zeros((100, 100), dtype=np.uint8)
image[20:40, 20:40] = 1
image[60:80, 60:80] = 2

# Adding Gaussian noise to the image
noisy_image = image.astype(np.float32) + np.random.normal(scale=0.5, size=image.shape)
noisy_image = np.clip(noisy_image, 0, 2).astype(np.uint8)

# Apply Tsutomu's thresholding
threshold_value = filters.threshold_otsu(noisy_image)

# Segment the image using the threshold
segmented_image = np.zeros_like(noisy_image)
segmented_image[noisy_image >= threshold_value] = 1

# Display the results
cv2.imshow('Original Image', (image * 127).astype(np.uint8))
cv2.imshow('Noisy Image', (noisy_image * 127).astype(np.uint8))
cv2.imshow('Segmented Image', (segmented_image * 127).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
