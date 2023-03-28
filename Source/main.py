import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import gray


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    # Dilation binary image
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation

    # Erosion binary image
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion

    # Opening binary image
    elif mor_op == 'opening':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening

    # Closing binary image
    elif mor_op == 'closing':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.closing(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing


    # Hit or miss binary image
    elif mor_op == 'hit':
        img_hit = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('OpenCV hit or miss image', img_hit)
        cv2.waitKey(wait_key_time)

        img_hit_manual = binary.hitOrMiss(img, kernel)
        cv2.imshow('manual hit or miss image', img_hit_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hit

    # Extract boundary in binary image
    elif mor_op == 'boundary':
        boundary = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('OpenCV boundary of image', boundary)
        cv2.waitKey(wait_key_time)

        boundary_manual = binary.boundaryEx(img, kernel)
        cv2.imshow('manual boundary of image', boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = boundary

    # Thinning binary image
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        thinning_manual = binary.thinning(img, kernel)
        cv2.imshow('manual thinning image', thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning

    # Dilation graysale image
    elif mor_op == 'Gdilate':
        img_dilation = cv2.dilate(img_gray, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = gray.dilate(img_gray, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation

    # Erosion grayscale image
    elif mor_op == 'Gerode':
        img_dilation = cv2.dilate(img_gray, kernel)
        cv2.imshow('OpenCV erosion image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = gray.dilate(img_gray, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation

    # Opening grayscale image
    elif mor_op == 'Gopening':
        img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = gray.opening(img_gray, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening

    # Closing grayscale image
    elif mor_op == 'Gclosing':
        img_closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = gray.closing(img_gray, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing

    # Extract boundary in grayscale image
    elif mor_op == 'Gboundary':
        boundary = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('OpenCV boundary of image', boundary)
        cv2.waitKey(wait_key_time)

        boundary_manual = gray.boundaryEx(img_gray, kernel)
        cv2.imshow('manual boundary of image', boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = boundary

    # Top hat in grayscale image
    elif mor_op == 'top':
        img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV top hat image', img_tophat)
        cv2.waitKey(wait_key_time)

        topHat_manual = gray.topHat(img_gray, kernel)
        cv2.imshow('manual top hat image', topHat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_tophat

    # Black hat in grayscale image
    elif mor_op == 'black':
        img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV black hat image', img_blackhat)
        cv2.waitKey(wait_key_time)

        blackHat_manual = gray.blackHat(img_gray, kernel)
        cv2.imshow('manual black hat image', blackHat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_blackhat

    if img_out is not None:
        cv2.imwrite(out_file, img_out)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
