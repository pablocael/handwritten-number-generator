from collections.abc import Iterable
from typing import Tuple, Iterable
import helpers
import numpy as np
from mnist import MNIST

class ImageGenerator:

    def generate_from_data(data: Iterable) -> np.ndarray:
        pass


class NumberSequenceGenerator(ImageGenerator):

    def __init__(self, spacing_range: Tuple[int, int], output_image_width: int):

        self._spacing_range = spacing_range
        self._output_image_width = output_image_width

    def _load_static_digits_dataset(self):

        data = MNIST('../data')
        images, labels = mndata.load_training()

    def generate_from_data(self, data: Iterable):
        pass



def generate_numbers_sequence(digits: Iterable[int], spacing_range: Tuple[int, int], image_width: int) -> np.ndarray:
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using a uniform distribution.

    Parameters
    ----------
    digits:
    An iterable containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
    a (minimum, maximum) int pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        specifies the width of the image in pixels.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
    1 (white), the first dimension corresponding to the height and the second
    dimension to the width.
    """
    print('aaa')
