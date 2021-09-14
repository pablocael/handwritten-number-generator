"""
Provides image processing hepler functions to aid number_generator module.
"""

import numpy as np
from PIL import Image
from typing import Tuple

def zero_pad_centered_axis(image: np.ndarray, axis: int, new_size: int):

    """
    Pads an image dimension in order to keep content centered in that dimension (or one pixel off in worst case)

    Parameter
    ---------
    image: numpy 2d ndarray representing an monochromatic image

    axis: the axis index representing the dimension to pad

    new_size: the total new size after padding - new_size must be bigger than image dimension at choosen axis
    """

    current_size = image.shape[axis]
    if current_size >= new_size:
        return image

    diff = (new_size - current_size)

    pad_size =  diff // 2

    # pad size is diff // 2
    pad_left = pad_size

    # pad right is the remaining pixels. this is needed because diff / 2 can have remaining of 1
    pad_right = new_size - current_size - pad_size

    # pad only given axis dimension

    dims_pad = [(0,0), (0,0)]
    dims_pad[axis] = (pad_left, pad_right)
    return np.pad(image, pad_width=dims_pad, mode='constant')

def rescale_to_width(image: np.ndarray, new_width) -> np.ndarray:
    """
    Rescales image width keeping height fixed.
    This happens by whether padding width with zeros on both sides, or rescaling width and padding height in both sides.
    Original image contents aspect ratio is not affected
    """

    original_width, original_height = image.shape[1], image.shape[0]
    if original_width == new_width:
        return image

    # if new_width is larger then input image, just pad it with background on both ends
    if original_width < new_width:
        return zero_pad_centered_axis(image, 1, new_width)

    # original_width > new_width
    # if output width is smaller, when we need to first rescale the image keeping aspect ratio,
    # then insert rescaled image into a self._output_image_width x original_height image with vertical centering

    scale_factor = new_width / original_width

    pil_image = Image.fromarray(image)

    scaled_width = int(np.round(pil_image.width * scale_factor))
    secaled_height = int(np.round(pil_image.height * scale_factor))
    pil_image = pil_image.resize((scaled_width, secaled_height), Image.LANCZOS)

    # now pad with zeros on y dimension to keep original height

    image = np.array(pil_image)
    return zero_pad_centered_axis(image, 0, original_height)

def calculate_binary_image_contents_bbox(binary_image: np.ndarray) -> Tuple[int, int, int, int]:
    """
        Extracts a bounding box from a binary image of format numpy.uint8, where relevant content has value > 0 (signal)
        and background has value 0.

        Parameters
        ----------

        binary_image:
        An numpy image assumed to have rows in the first dimension (axis=0), and columns in
        the second dimension (axis=1), so binary_image[y,x] retrievens the pixel (x,y).
        The image typ is numpy.uint8 and should have background values set to 0 and signal values set to non zero.

        Returns
        --------
        The function returns a tuple (xmin, ymin, xmax, ymax) that delimits values of the interest region (signal).
        xmin: is the x-axis index of the first pixel with relevant value among all rows
        ymin: is the y-axis index of the first pixel with relevant value among all columns
        xmax: is the x-axis

    """

    # find rows and columns with any relevant data on it
    rows = np.any(binary_image, axis=0)
    cols = np.any(binary_image, axis=1)

    # trunk data to intervals with relevant data (remove background)
    # where() function without argument returns only indexes of pixels with relevant values and clamp background ones
    rows = np.where(rows)[0]
    cols = np.where(cols)[0]
    if rows.size == 0 or cols.size == 0:
        # an empty image (background only)
        return (0, 0, 0, 0)

    # get first and last index found by where() function for each row and col
    rmin, rmax = rows[[0, -1]]
    cmin, cmax = cols[[0, -1]]

    return (rmin, cmin, rmax, cmax)
