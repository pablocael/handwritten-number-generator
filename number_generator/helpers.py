"""
Provides image processing hepler functions to aid number_generator module.
"""

from typing import Tuple
import numpy as np


def calculate_bbox_from_binary_image(binary_image: np.array) -> Tuple[int, int, int, int]:
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
