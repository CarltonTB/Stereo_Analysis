import unittest
import cv2
import numpy as np
import stereo_analysis as sa


class StereoAnalysisTest(unittest.TestCase):

    def test_region_based_analysis_SAD(self):
        image1 = cv2.imread('sample_images/cones/im2.png')
        image2 = cv2.imread('sample_images/cones/im6.png')
        disparity_map = sa.region_based_analysis_SAD(image1, image2, (7, 7), 50)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_region_based_analysis_SSD(self):
        image1 = cv2.imread('sample_images/cones/im2.png')
        image2 = cv2.imread('sample_images/cones/im6.png')
        disparity_map = sa.region_based_analysis_SSD(image1, image2, (7, 7), 50)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_region_based_analysis_NCC(self):
        image1 = cv2.imread('sample_images/cones/im2.png')
        image2 = cv2.imread('sample_images/cones/im6.png')
        disparity_map = sa.region_based_analysis_NCC(image1, image2, (7, 7), 50)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
