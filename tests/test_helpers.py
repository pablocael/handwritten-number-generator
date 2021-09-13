from number_generator import helpers
import numpy as np

def test_calculate_binary_image_contents_bbox():

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

def test_zero_pad_centered_axis():

    # test non divisible by two width padding
    output_width = 111
    input_width = 50
    input_height = 28

    input_image = np.ones((28, input_width))

    result = helpers.zero_pad_centered_axis(input_image, 1, output_width)
    assert result.shape[1] == output_width

    # assert the we pad zeros on the left and on the right
    # since image is all ones, we can check padding
    # lets use contents bbox detector for checking
    x0, y0, x1, y1 = helpers.calculate_binary_image_contents_bbox(result)
    assert (x1 - x0)+1 == input_width

