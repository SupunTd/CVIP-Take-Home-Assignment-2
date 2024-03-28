import numpy as np
import cv2


def region_growing(image, seed, threshold):
    visited = set()
    queue = [seed]
    segmented_image = np.zeros_like(image)

    while queue:
        current_pixel = queue.pop(0)
        segmented_image[current_pixel] = 255
        visited.add(tuple(current_pixel))

        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (current_pixel[0] + i, current_pixel[1] + j)
                if neighbor not in visited and np.abs(image[current_pixel] - image[neighbor]) <= threshold:
                    queue.append(neighbor)
                    visited.add(neighbor)

    return segmented_image


# Generate synthetic image with 2 objects and background
image = np.zeros((100, 100), dtype=np.uint8)
image[20:40, 20:40] = 100  # Object 1
image[60:80, 60:80] = 200  # Object 2

# Define seed point inside the object of interest
seed = (25, 25)

# Set threshold for region growing
threshold = 20

# Apply region growing segmentation
segmented_image = region_growing(image, seed, threshold)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
