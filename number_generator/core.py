from collections.abc import Iterable
import logging
from typing import Iterable, Tuple

import numpy as np

from mnist import MNIST

from . import helpers

logger = logging.getLogger(__name__)


class DigitImageDataset:
    """
        DigitImageDataset class.
        Stores image samples for digits in interval [0,9].
    """
    def __init__(self, labels: np.array, images: np.ndarray):
        """
            Allow constructing digit dataset from plain samples and labels

            Parameters
            ----------

            labels:
            a np.array of type uint8 storing each image digit sample label

            images:
            digit samples in a numpy array with format (N, height, width), where N is the number of samples


            labels length and images.shape[0] must have same size
        """

        # create a dictionary to store classes examples
        self._digit_samples = {}
        for i in range(10):
            mask = labels == i
            self._digit_samples[i] = images[mask]

        # store sample image shape as (width, height)
        self._sample_shape = images.shape[1:][::-1]


    def get_sample(self, digit: int) -> np.ndarray:
        return self._digit_samples[digit]

    def get_sample_shape(self):
        return self._sample_shape


class ImageGenerator:

    def generate_from_data(data: Iterable) -> np.ndarray:
        pass


class DigitSequenceImageGenerator(ImageGenerator):

    """
        NumberSequenceGenerator class

        Inherits from ImageGenerator.

        Generates synthetic digits by randomly sampling from a given digit dataset separated by digit class.

    """
    def __init__(self, dataset: DigitImageDataset, spacing_range: Tuple[int, int], output_image_width: int):

        self._dataset = dataset
        self._spacing_range = spacing_range
        self._output_image_width = output_image_width

    def _generate_blank_block(self, width, height):
        return np.zeros((height, width))

    def generate_from_data(self, data: Iterable):
        pass


def load_mnist_digit_dataset():

    data = MNIST(path='./data/', return_type='numpy', gz=True)
    test_imgs, test_labels = data.load_testing()
    train_imgs, train_labels = data.load_training()

    # gather all images from both train and test
    images = np.concatenate([test_imgs, train_imgs], axis=0)
    labels = np.concatenate([test_labels, train_labels], axis=0)

    # assume images have equal dimensions in 0 and 1 axis
    image_dim_size = int(np.sqrt(images.shape[1]))

    # reshape to square dimension instead of single array data
    images = np.uint8(images).reshape(-1, image_dim_size, image_dim_size)

    return DigitImageDataset(labels, images)


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
    # Create the dataset using MNIST
    mnist_dataset = load_mnist_digit_dataset()

    # Generate the synthetic sequence using the digit dataset created
    digit_sequence_generator = DigitSequenceImageGenerator(mnist_dataset, spacing_range=spacing_range, output_image_width=image_width)

    return digit_sequence_generator.generate_from_data(digits)


