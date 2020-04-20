import unittest
import cv2
import numpy as np
import region_based_analysis as rba
import feature_based_analysis as fba
import multi_resolution_analysis as mra
import image_blending as ib


class StereoAnalysisTest(unittest.TestCase):

    def test_region_based_analysis_SAD(self):
        # image1 = cv2.imread('sample_images/cones/im6.png')
        # image2 = cv2.imread('sample_images/cones/im2.png')
        # disparity_map = rba.region_based_analysis_sad(image1, image2, (7, 7), 50, search_direction="R")
        # image1 = cv2.imread('sample_images/cones/im6.png')
        # image2 = cv2.imread('sample_images/cones/im2.png')
        # disparity_map = rba.region_based_analysis_sad(image2, image1, (7, 7), 50, search_direction="L")
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        disparity_map = rba.region_based_analysis_sad(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map rb sad', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_region_based_analysis_SSD(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        disparity_map = rba.region_based_analysis_ssd(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map rb ssd', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_region_based_analysis_NCC(self):
        image1 = cv2.imread('sample_images/cones/im2.png')
        image2 = cv2.imread('sample_images/cones/im6.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        disparity_map = rba.region_based_analysis_ncc(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map rb ncc', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_feature_based_analysis_descriptor_value(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        starting_disparity_map = rba.region_based_analysis_sad(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = fba.feature_based_analysis_descriptor_value(image1, image2, (7, 7), 50,
                                                                    starting_disparity_map, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map fb descriptor value', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_feature_based_analysis_sad(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        starting_disparity_map = rba.region_based_analysis_sad(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = fba.feature_based_analysis_sad(image1, image2, (7, 7), 50,
                                                       starting_disparity_map, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map fb sad', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_feature_based_analysis_ssd(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        starting_disparity_map = rba.region_based_analysis_ssd(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = fba.feature_based_analysis_ssd(image1, image2, (7, 7), 50,
                                                       starting_disparity_map, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map fb ssd', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_feature_based_analysis_ncc(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        starting_disparity_map = rba.region_based_analysis_ncc(image1, image2, (7, 7), 50, search_direction="BOTH")
        disparity_map = fba.feature_based_analysis_ncc(image1, image2, (7, 7), 50,
                                                                  starting_disparity_map, search_direction="BOTH")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # For visual debugging:
        cv2.imshow('disparity map fb ncc', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_multi_resolution_analysis_sad(self):
        # image2 = cv2.imread('sample_images/cones/im6.png')
        # image1 = cv2.imread('sample_images/cones/im2.png')
        image2 = cv2.imread('sample_images/teddy/im6.png')
        image1 = cv2.imread('sample_images/teddy/im2.png')
        disparity_map = mra.multi_resolution_analysis(image2, image1, (7, 7), 50, 3,
                                                      search_both=True, matching_score="SAD")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # cv2.imwrite('results/region_based/cones/multi-resolution7x7template50windowSADn=3.png', disparity_map)
        cv2.imwrite('results/region_based/teddy/multi-resolution7x7template50windowSADn=3.png', disparity_map)
        # For visual debugging:
        # cv2.imshow('disparity map mr rb sad n=3', disparity_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_multi_resolution_analysis_ssd(self):
        # image2 = cv2.imread('sample_images/cones/im6.png')
        # image1 = cv2.imread('sample_images/cones/im2.png')
        image2 = cv2.imread('sample_images/teddy/im6.png')
        image1 = cv2.imread('sample_images/teddy/im2.png')
        disparity_map = mra.multi_resolution_analysis(image2, image1, (7, 7), 50, 3,
                                                      search_both=True, matching_score="SSD")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        # cv2.imwrite('results/region_based/cones/multi-resolution7x7template50windowSSDn=3.png', disparity_map)
        cv2.imwrite('results/region_based/teddy/multi-resolution7x7template50windowSSDn=3.png', disparity_map)
        # # For visual debugging:
        # cv2.imshow('disparity map mr rb ssd n=3', disparity_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_multi_resolution_analysis_ncc(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        # image2 = cv2.imread('sample_images/teddy/im6.png')
        # image1 = cv2.imread('sample_images/teddy/im2.png')
        disparity_map = mra.multi_resolution_analysis(image1, image2, (7, 7), 50, 3,
                                                      search_both=True, matching_score="NCC")
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        cv2.imwrite('results/region_based/cones/multi-resolution7x7template50windowNCCn=3.png', disparity_map)
        # cv2.imwrite('results/region_based/teddy/multi-resolution7x7template50windowNCCn=3.png', disparity_map)
        # For visual debugging:
        # cv2.imshow('disparity map mr rb ncc n=3', disparity_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_multi_resolution_feature_based_sad(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        disparity_map = mra.multi_resolution_analysis(image2, image1, (7, 7), 50, 3,
                                                      search_both=True, matching_score="SAD", feature_based=True)
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        cv2.imwrite('results/feature_based/cones/multi-resolution7x7template50windowSADn=3.png', disparity_map)

    def test_multi_resolution_feature_based_ssd(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        disparity_map = mra.multi_resolution_analysis(image2, image1, (7, 7), 50, 3,
                                                      search_both=True, matching_score="SSD", feature_based=True)
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        cv2.imwrite('results/feature_based/cones/multi-resolution7x7template50windowSSDn=3.png', disparity_map)

    def test_multi_resolution_feature_based_ncc(self):
        image2 = cv2.imread('sample_images/cones/im6.png')
        image1 = cv2.imread('sample_images/cones/im2.png')
        disparity_map = mra.multi_resolution_analysis(image2, image1, (7, 7), 50, 3,
                                                      search_both=True, matching_score="NCC", feature_based=True)
        disparity_map = rba.normalize_disparity_values(disparity_map)
        self.assertEqual(np.size(disparity_map, 0), np.size(image1, 0))
        self.assertEqual(np.size(disparity_map, 0), np.size(image2, 0))
        self.assertEqual(np.size(disparity_map, 1), np.size(image1, 1))
        self.assertEqual(np.size(disparity_map, 1), np.size(image2, 1))
        cv2.imwrite('results/feature_based/cones/multi-resolution7x7template50windowNCCn=3.png', disparity_map)


if __name__ == '__main__':
    unittest.main()
