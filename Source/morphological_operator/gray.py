import numpy as np
import cv2

# Dilation
def dilate(img, kernel):
    # Add pixels around image to avoid loss information when dilation
    img = np.pad(img, ((kernel.shape[0] // 2),(kernel.shape[1] // 2)), 'constant', constant_values=(0,0))
    # Create 0 matrix to save result
    dilated = np.zeros_like(img)

    # Implement erosion
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            dilated[i,j] = np.max(img[i-1:i+2, j-1:j+2] * kernel)

    # Reize of dilated image = original image
    dilated = dilated[1:-1, 1:-1]

    return dilated

# Erosion
def erode(img, kernel):
    # Add pixels around image to avoid loss information when erosion
    img = np.pad(img, ((kernel.shape[0] // 2),(kernel.shape[1] // 2)), 'constant', constant_values=(0,0))

    # Create 0 matrix to save result
    eroded = np.zeros_like(img)

    # Implement erosion
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            eroded[i,j] = np.min(img[i-1:i+2, j-1:j+2] * kernel)

    # Reize of eroded image = original image
    eroded = eroded[1:-1, 1:-1]

    return eroded

# Opening
def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)

# Closing
def closing(img, kernel):
    return erode(dilate(img, kernel), kernel)

# Gradient
def boundaryEx(img, kernel):
    return dilate(img, kernel) - erode(img, kernel)

# Top-hat
def topHat(img, kernel):
    return img - opening(img, kernel)

# Black-hat
def blackHat(img, kernel):
    return closing(img, kernel) - img

