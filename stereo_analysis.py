# Author: Carlton Brady
import cv2
import numpy as np
import sys
import image_blending as ib


def region_based_analysis_SAD(image1, image2, template_size, window_size, num_levels=0):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param num_levels: the number of levels in the image pyramid to be used for multi-resolution analysis
    :return: the disparity map result from the analysis of the two images using the SAD matching score
    """
    # images should be of the same size
    assert np.size(image1) == np.size(image2)
    image1_greyscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_greyscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    assert np.size(image1_greyscale) == np.size(image2_greyscale)
    image_height = np.size(image1_greyscale, 0)
    image_width = np.size(image1_greyscale, 1)
    template_width = template_size[0]
    template_height = template_size[1]
    template_pixels = template_height*template_width
    disparity_map = np.zeros((image_height, image_width), dtype=np.float32)
    # for every pixel where a full template chunk can be taken
    for i in range(0, image_height-template_height):
        for j in range(0, image_width-template_width):
            minSADscore = sys.maxsize
            minSADlocation = None
            template = image1_greyscale[i:i+template_height, j:j+template_width]
            # subtrack the templates mean from it
            template_mean = np.sum(template) / template_pixels
            template = template - template_mean
            left_boundary = j - window_size if j >= window_size else 0
            right_boundary = j + window_size if j + window_size + template_width < image_width else image_width - template_width
            distance = 0
            for k in range(left_boundary, right_boundary):
                image2_matrix = image2_greyscale[i:i+template_height, k:k+template_width]
                # subtract the matrix's mean from it
                image2_matrix_mean = np.sum(image2_matrix) / template_pixels
                image2_matrix = image2_matrix - image2_matrix_mean
                SAD_matrix = abs(template - image2_matrix)
                total_SAD = np.sum(SAD_matrix)
                if total_SAD < minSADscore:
                    minSADscore = total_SAD
                    minSADlocation = distance
                distance += 1
            disparity_value = abs(window_size - minSADlocation)
            disparity_matrix = np.zeros((template_height, template_width))
            disparity_matrix.fill(disparity_value)
            disparity_map[i:i+template_height, j:j+template_width] = disparity_matrix
            # print("i:", i)
            # print("j:", j)
    # Normalize disparity values
    largest = disparity_map.max()
    increment = 255 / largest
    disparity_map = (255 - disparity_map) * increment
    disparity_map = np.rint(disparity_map)
    disparity_map = disparity_map.astype(np.uint8)
    disparity_map = cv2.bitwise_not(disparity_map)
    return disparity_map


def region_based_analysis_SSD(image1, image2, template_size, window_size, num_levels=0):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param num_levels: the number of levels in the image pyramid to be used for multi-resolution analysis
    :return: the disparity map result from the analysis of the two images using the SSD matching score
    """
    # images should be of the same size
    assert np.size(image1) == np.size(image2)
    image1_greyscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_greyscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    assert np.size(image1_greyscale) == np.size(image2_greyscale)
    image_height = np.size(image1_greyscale, 0)
    image_width = np.size(image1_greyscale, 1)
    template_width = template_size[0]
    template_height = template_size[1]
    template_pixels = template_height*template_width
    disparity_map = np.zeros((image_height, image_width), dtype=np.float32)
    # for every pixel where a full template chunk can be taken
    for i in range(0, image_height-template_height):
        for j in range(0, image_width-template_width):
            minSSDscore = sys.maxsize
            minSSDlocation = None
            template = image1_greyscale[i:i+template_height, j:j+template_width]
            # subtrack the templates mean from it
            template_mean = np.sum(template) / template_pixels
            template = template - template_mean
            left_boundary = j - window_size if j >= window_size else 0
            right_boundary = j + window_size if j + window_size + template_width < image_width else image_width - template_width
            distance = 0
            for k in range(left_boundary, right_boundary):
                image2_matrix = image2_greyscale[i:i+template_height, k:k+template_width]
                # subtract the matrix's mean from it
                image2_matrix_mean = np.sum(image2_matrix) / template_pixels
                image2_matrix = image2_matrix - image2_matrix_mean
                SSD_matrix = (template-image2_matrix)**2
                total_SSD = np.sum(SSD_matrix)
                if total_SSD < minSSDscore:
                    minSSDscore = total_SSD
                    minSSDlocation = distance
                distance += 1
            disparity_value = abs(window_size - minSSDlocation)
            disparity_matrix = np.zeros((template_height, template_width))
            disparity_matrix.fill(disparity_value)
            disparity_map[i:i+template_height, j:j+template_width] = disparity_matrix
            # print("i:", i)
            # print("j:", j)
    # Normalize disparity values
    largest = disparity_map.max()
    increment = 255 / largest
    disparity_map = (255 - disparity_map) * increment
    disparity_map = np.rint(disparity_map)
    disparity_map = disparity_map.astype(np.uint8)
    disparity_map = cv2.bitwise_not(disparity_map)
    return disparity_map


def region_based_analysis_NCC(image1, image2, template_size, window_size, num_levels=0):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param num_levels: the number of levels in the image pyramid to be used for multi-resolution analysis
    :return: the disparity map result from the analysis of the two images using the NCC matching score
    """
    # images should be of the same size
    assert np.size(image1) == np.size(image2)
    image1_greyscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_greyscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    assert np.size(image1_greyscale) == np.size(image2_greyscale)
    image_height = np.size(image1_greyscale, 0)
    image_width = np.size(image1_greyscale, 1)
    template_width = template_size[0]
    template_height = template_size[1]
    template_pixels = template_height*template_width
    disparity_map = np.zeros((image_height, image_width), dtype=np.float32)
    # for every pixel where a full template chunk can be taken
    for i in range(0, image_height-template_height):
        for j in range(0, image_width-template_width):
            maxNCCscore = -1
            maxNCClocation = None
            template = image1_greyscale[i:i+template_height, j:j+template_width]
            # subtrack the templates mean from it
            template_mean = np.sum(template) / template_pixels
            template = template - template_mean
            left_boundary = j - window_size if j >= window_size else 0
            right_boundary = j + window_size if j + window_size + template_width < image_width else image_width - template_width
            distance = 0
            for k in range(left_boundary, right_boundary):
                image2_matrix = image2_greyscale[i:i+template_height, k:k+template_width]
                # subtract the matrix's mean from it
                image2_matrix_mean = np.sum(image2_matrix) / template_pixels
                image2_matrix = image2_matrix - image2_matrix_mean
                # total_NCC = np.sum(template*image2_matrix)/(np.std(template)*np.std(image2_matrix))
                denom = (np.sum(template**2)*np.sum(image2_matrix**2))**0.5
                total_NCC = np.sum(template*image2_matrix)/denom
                assert(-1 <= total_NCC <= 1)
                if total_NCC > maxNCCscore:
                    maxNCCscore = total_NCC
                    maxNCClocation = distance
                distance += 1
            disparity_value = abs(window_size - maxNCClocation)
            disparity_matrix = np.zeros((template_height, template_width))
            disparity_matrix.fill(disparity_value)
            disparity_map[i:i+template_height, j:j+template_width] = disparity_matrix
            # print("i:", i)
            # print("j:", j)
    # Normalize disparity values
    largest = disparity_map.max()
    increment = 255 / largest
    disparity_map = (255 - disparity_map) * increment
    disparity_map = np.rint(disparity_map)
    disparity_map = disparity_map.astype(np.uint8)
    disparity_map = cv2.bitwise_not(disparity_map)
    return disparity_map


# image1 = cv2.imread('sample_images/cones/im2.png')
# image2 = cv2.imread('sample_images/cones/im6.png')
# disparity_map = region_based_analysis_SAD(image1, image2, (5, 5), 50)
# # disparity_map = region_based_analysis_SSD(image1, image2, (5, 5), 50)
# # disparity_map = region_based_analysis_NCC(image1, image2, (5, 5), 50)
# cv2.imshow('disparity map', disparity_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()