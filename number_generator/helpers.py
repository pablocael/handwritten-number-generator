
def bbox_from_binary_image(binary_image):
    """
        Extracts a bounding box from a binary image of format numpy.uint8, where relevant content has value > 0 and
        background has value 0. The image is assumed to have rows in the first dimension (axis=0), and columns in the second
        dimension (axis=1).

        return:
        The function returns (xmin, ymin, xmax, ymax) values of the interest region.
        xmin: is the x-axis index of the first pixel with relevant value among all rows
        ymin: is the y-axis index of the first pixel with relevant value among all columns
        xmax: is the x-axis

    """

    # find rows and columns with any relevant data on it
    rows = np.any(binary_image, axis=0)
    cols = np.any(binary_image, axis=1)

    # trunk data to intervals with relevant data (remove background)
    rows = np.where(rows)[0]
    cols = np.where(cols)[0]

    rmin, rmax = rows[[0, -1]]
    cmin, cmax = cols[[0, -1]]

    return rmin, cmin, rmax, cmax
