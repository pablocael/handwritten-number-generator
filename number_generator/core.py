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
    Stores image examples for digits in interval [0,9].
    """
    def __init__(self, labels: np.array, images: np.ndarray):
        """
        Construct a digit dataset from list of examples and labels

        Parameters
        ----------

        labels:
        a np.array of type uint8 storing each image digit example label

        images:
        digit examples in a numpy array with format (N, height, width), where N is the number of examples


        labels length and images.shape[0] must have same size
        """

        # create a dictionary to store classes examples
        self._digit_examples = {}
        for i in range(10):
            mask = labels == i
            self._digit_examples[i] = images[mask]

        # store sample image shape as (width, height)
        self._sample_shape = images.shape[1:][::-1]


    def get_digit_examples(self, digit: int) -> np.ndarray:
        """
        Retrieve all examples for the given digit
        """

        return self._digit_examples[digit]

    def random_sample_digit(self, digit: int) -> np.ndarray:
        """
       Sample an example from given digit class using a discrete uniform distribution
        """

        index = np.random.randint(self._digit_examples[digit].shape[0])
        return self._digit_examples[digit][index]

    def get_example_shape(self) -> Tuple[int, int]:
        return self._sample_shape

class ImageGenerator:

    def generate_from_data(data: Iterable) -> np.ndarray:
        """
        To be reimplemented by concrete implementations of ImageGenerator.

        Generate a synthetic image from an iterable input data.
        """
        raise NotImplementedError

class DigitSequenceImageGenerator(ImageGenerator):

    """
    NumberSequenceGenerator class

    Inherits from ImageGenerator.

    Generates synthetic digits by randomly sampling from a given digit dataset separated by digit class.
    """
    def __init__(self, dataset: DigitImageDataset, spacing_range: Tuple[int, int], output_image_width: int):

        """
        Construct a DigitSequenceGenerator class using the input digit dataset.

        Parameters
        ----------
        dataset:
        A generic digit dataset to retrieve digits images from.

        spacing_range:
        a (minimum, maximum) int pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.

        output_image_width:
        specifies the width of the image in pixels.

        Returns
        -------
        The image containing the sequence of numbers. Images should be represented
        as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
        1 (white), the first dimension corresponding to the height and the second
        dimension to the width.
        """

        self._dataset = dataset
        self._spacing_range = spacing_range
        self._output_image_width = output_image_width

    def _generate_blank_block(self, width: int, height: int) -> np.ndarray:
        return np.zeros((height, width))

    def _generate_random_space_block(self) -> np.ndarray:
        """
        Generate a blank image block with fixed height and width sampled from a discrete$ uniform
        distribution.
        """

        _, height = self._dataset.get_example_shape()

        space_width = np.random.randint(self._spacing_range[0], self._spacing_range[1]+1)
        return self._generate_blank_block(space_width, height)

    def generate_from_data(self, digits: Iterable):

        N = len(digits) # number of total digits in the sequence

        # stores the intermediate generate sequence of images that corresponds to
        # each digit in input sequence plus spacer images
        intermediate_result_images = []

        for index, digit in enumerate(digits):

            # random sample a digit
            digit_image = self._dataset.random_sample_digit(digit)

            # get relevant content bounding box
            x0, _, x1, _ = helpers.calculate_binary_image_contents_bbox(digit_image)

            # crop digit contents on x axis
            digit_image = digit_image[:,x0:x1]

            intermediate_result_images.append(digit_image)

            # add spacing if not last element
            if index == N-1:
                continue

            random_spacing_block = self._generate_random_space_block()
            intermediate_result_images.append(random_spacing_block)


        return np.hstack(intermediate_result_images)

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


