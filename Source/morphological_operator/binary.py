import numpy as np
import cv2

def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    img_shape = img.shape
    eroded_img = np.zeros(img_shape, dtype=np.uint8)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0] - 1
            j_ = j + kernel.shape[1] - 1
            if i_ >= img_shape[0] or j_ >= img_shape[1]:
                continue
            if kernel_ones_count == (kernel * img[i:i_+1, j:j_+1]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 255
    return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''


def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    img_shape = img.shape
    dil_img = np.zeros(img_shape, dtype=np.uint8)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0] - 1
            j_ = j + kernel.shape[1] - 1
            if i_ >= img_shape[0] or j_ >= img_shape[1]:
                continue
            if (kernel * img[i:i_+1, j:j_+1]).sum() > 0:
                dil_img[i:i_+1, j:j_+1] = 255
    return dil_img[:img_shape[0], :img_shape[1]]

def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)

def closing(img, kernel):
    img_dilte = dilate(img, kernel)
    img_er = erode(img_dilte, kernel)

    return  img_er

def hitOrMiss(img, kernel):
    img_c =  255- img
    kernel_c = 255 - kernel

    erode_img = erode(img, kernel)
    erode_img_c = erode(img_c, kernel_c)
    hit_or_miss_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if erode_img[i,j] == 255 and erode_img_c[i,j] == 0:
                hit_or_miss_img[i,j] = 255

    return hit_or_miss_img

def boundaryEx(img, kernel):
    return img - erode(img, kernel)

def thinning(img, kernel):
    while 1:
        last_img = img.copy()

        img = img - opening(img, kernel) + erode(img, kernel)

        if np.all(img == last_img):
            break
    return img

