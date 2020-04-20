# Author: Carlton Brady
import cv2
import numpy as np
import region_based_analysis as rba
import feature_based_analysis as fba
import multi_resolution_analysis as mra
import image_blending as ib


# Note: multi-resolution with 3 levels takes ~20 minutes on my machine, and 1 level takes ~10 minutes

def run_region_based_demo():
    matching_score_choice = input("Matching score options (enter the corresponding number for the desired score):\n"
                   "1) SAD\n"
                   "2) SSD\n"
                   "3) NCC\n")
    template_height = input("Enter desired template height in pixels (7 works well)\n")
    template_width = input("Enter desired template width in pixels (7 works well)\n")
    search_window = input("Enter desired search window in pixels (50 works well)\n")
    pyramid_levels = input("Enter desired number of pyramid levels (1-3 work best)\n")
    matching_score_choice = int(matching_score_choice)
    template_height = int(template_height)
    template_width = int(template_width)
    search_window = int(search_window)
    pyramid_levels = int(pyramid_levels)
    print("Running Region-based analysis on cones images...")
    image2 = cv2.imread('sample_images/cones/im6.png')
    image1 = cv2.imread('sample_images/cones/im2.png')
    if pyramid_levels == 1:
        if matching_score_choice == 1:
            disparity_map = rba.region_based_analysis_sad(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
        elif matching_score_choice == 2:
            disparity_map = rba.region_based_analysis_ssd(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
        elif matching_score_choice == 3:
            disparity_map = rba.region_based_analysis_ncc(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
        else:
            print("Please enter a valid matching score number")
            return None
    else:
        if matching_score_choice == 1:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="SAD")
        elif matching_score_choice == 2:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="SSD")
        elif matching_score_choice == 3:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="NCC")
        else:
            print("Please enter a valid matching score number")
            return None
    disparity_map = rba.normalize_disparity_values(disparity_map)
    cv2.imshow('disparity', disparity_map)
    print("Click on the image and press any key to close the image and end the demo")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Demo Complete")


def run_feature_based_demo():
    matching_score_choice = input("matching score options (enter the corresponding number for the desired score):\n"
                                  "1) SAD\n"
                                  "2) SSD\n"
                                  "3) NCC\n"
                                  "4) Descriptor Value\n")
    template_height = input("Enter desired template height in pixels (7 seems to work well)\n")
    template_width = input("Enter desired template width in pixels (7 seems to work well)\n")
    search_window = input("Enter desired search window in pixels (50 seems to work well)\n")
    pyramid_levels = input("Enter desired number of pyramid levels (1-3 work best)\n")
    matching_score_choice = int(matching_score_choice)
    template_height = int(template_height)
    template_width = int(template_width)
    search_window = int(search_window)
    pyramid_levels = int(pyramid_levels)
    print("Running Feature-based analysis on cones images...")
    image2 = cv2.imread('sample_images/cones/im6.png')
    image1 = cv2.imread('sample_images/cones/im2.png')
    if pyramid_levels == 1:
        if matching_score_choice == 4:
            init_map = rba.region_based_analysis_sad(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
            disparity_map = fba.feature_based_analysis_descriptor_value(image1, image2, (template_width, template_height),
                                                          search_window, init_map, search_direction="BOTH")
        elif matching_score_choice == 1:
            init_map = rba.region_based_analysis_sad(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
            disparity_map = fba.feature_based_analysis_sad(image1, image2, (template_width, template_height),
                                                          search_window, init_map, search_direction="BOTH")
        elif matching_score_choice == 2:
            init_map = rba.region_based_analysis_ssd(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
            disparity_map = fba.feature_based_analysis_ssd(image1, image2, (template_width, template_height),
                                                          search_window, init_map, search_direction="BOTH")
        elif matching_score_choice == 3:
            init_map = rba.region_based_analysis_ncc(image1, image2, (template_width, template_height),
                                                          search_window, search_direction="BOTH")
            disparity_map = fba.feature_based_analysis_ncc(image1, image2, (template_width, template_height),
                                                          search_window, init_map, search_direction="BOTH")
        else:
            print("Please enter a valid matching score number")
            return None
    else:
        if matching_score_choice == 1:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="SAD", feature_based=True)
        elif matching_score_choice == 2:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="SSD", feature_based=True)
        elif matching_score_choice == 3:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="NCC", feature_based=True)
        elif matching_score_choice == 4:
            disparity_map = mra.multi_resolution_analysis(image2, image1, (template_width, template_height),
                                                          search_window, pyramid_levels, search_both=True, matching_score="SAD", feature_based=True)
        else:
            print("Please enter a valid matching score number")
            return None
    disparity_map = rba.normalize_disparity_values(disparity_map)
    cv2.imshow('disparity', disparity_map)
    print("Click on the image and press any key to close the image and end the demo")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Demo Complete")


if __name__ == "__main__":
    demo_choice = input("Demo options (enter the corresponding number for the desired demo):\n"
                   "1) Region-based\n"
                   "2) Feature-based\n")
    if demo_choice == "1":
        run_region_based_demo()
    elif demo_choice == "2":
        run_feature_based_demo()
    else:
        print("please enter a value demo number")
