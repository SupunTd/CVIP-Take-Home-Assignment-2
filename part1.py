import numpy as np
import cv2

# Generate synthetic image with 2 objects and background
image = np.zeros((100, 100), dtype=np.uint8)
image[20:40, 20:40] = 100  # Object 1
image[60:80, 60:80] = 200  # Object 2

# Add Gaussian noise
noisy_image = image + np.random.normal(loc=0, scale=15, size=image.shape).astype(np.uint8)


# Implement Tsutomu's algorithm
_, thresholded_image = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Threshold Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
