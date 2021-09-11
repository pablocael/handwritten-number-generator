from number_generator import helpers
import numpy as np

def test_helpers_calculate_binary_image_contents_bbox():

    # create an empty image
    empty_image = np.zeros((28, 28), dtype=np.uint8)
    bbox = helpers.calculate_binary_image_contents_bbox(empty_image)

    # bbox of empty image (background only) should be all zeros
    assert bbox == (0, 0, 0, 0)

    # create a image with only two pixels defining the interest region
    simple_bounds =  np.zeros((50, 50), dtype=np.uint8)
    simple_bounds[10,12] = 50
    simple_bounds[45,42] = 200


    bbox = helpers.calculate_binary_image_contents_bbox(simple_bounds)

    # bbox of empty image (background only) should be all zeros
    assert bbox == (12, 10, 42, 45)

    # create a image with only two pixels defining the whole image region
    simple_bounds =  np.zeros((100, 100), dtype=np.uint8)
    simple_bounds[0, 0] = 50
    simple_bounds[99, 99] = 200


    bbox = helpers.calculate_binary_image_contents_bbox(simple_bounds)

    # bbox of empty image (background only) should be all zeros
    assert bbox == (0, 0, 99, 99)

