# Author: Carlton Brady
import cv2
import numpy as np


def region_based_analysis(image1, image2, template_size):
    """
    :param image1: an image of varying size
    :param image2: an image of varying size that is a view of the same scene with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :return: the disparity map result from the analysis of the two images
    """
    return None