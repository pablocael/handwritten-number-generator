import os
import sys
from collections.abc import Iterable
import logging
from typing import Iterable, Tuple
import pickle
import numpy as np

from PIL import Image

from . import helpers

logger = logging.getLogger(__name__)

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# will be loaded on demand, first time needed

this._DIGITS_DATASET_FILEPATH = os.path.join(sys.prefix, 'data/digits-dataset.pickle')
this._DIGITS_DATASET = None

class GenericDataset:

    def __init__(self, labels=None, images=None, metadata=None):
        """
        A generic dataset that can store and retrieve examples and labels
        """

        self._labels = labels or []
        self._images = images or []
        self._metadata = metadata or {}

    def save(self, output_filepath, metadata=None):
        data = {
            'labels': self._labels,
            'images': self._images,
            'metadata': metadata
        }
        with open(output_filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, input_filepath):
        data = None
        with open(input_filepath, 'rb') as handle:
            data = pickle.load(handle)

        if 'labels' not in data or 'images' not in data:
            raise Exception(f'loaded dataset at {input_filepath} is not a valid GenericDataset')

        self._labels = data['labels']
        self._images = data['images']

        self._metadata = {}
        if 'metadata' in data:
            self._metadata = data['metadata']

    def get_metadata(self):
        return self._metadata

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self._images[index], self._labels[index]


class DigitImageDataset(GenericDataset):
    """
    Stores image examples for digits in interval [0,9].
    """
    def __init__(self, labels: np.array = None, images: np.ndarray = None):
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
        if labels is None or images is None:
            return

        super(DigitImageDataset, self).__init__(labels=labels, images=images)
        self._initialize_digits_set(labels=labels, images=images)

    def _initialize_digits_set(self, labels: np.array, images: np.ndarray):

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

        if digit not in self._digit_examples:
            return np.ndarray()

        return self._digit_examples[digit]

    def random_sample_digit(self, digit: int) -> np.ndarray:
        """
       Sample an example from given digit class using a discrete uniform distribution
        """

        index = np.random.randint(self._digit_examples[digit].shape[0])
        return self._digit_examples[digit][index]

    def get_example_shape(self) -> Tuple[int, int]:
        return self._sample_shape

    def load(self, input_filepath):
        super(DigitImageDataset, self).load(input_filepath)

        self._initialize_digits_set(labels=self._labels, images=self._images)

class DigitImageDatasetAugmentator(DigitImageDataset):
    """
    A special digit dataset that will augment the data by using digit deformation
    """
    __accepted_augmentation_methods = [
        'elastic',
        'affine'
    ]
    def __init__(self, labels: np.array, images: np.ndarray, percent_augmentation: float, augmentation_method='elastic'):

        if augmentation_method not in DigitImageDatasetAugmentator.__accepted_augmentation_methods:
            raise ValueError(f'augmentation_method must be one of the following: {DigitImageDatasetAugmentator.__accepted_augmentation_methods}')

        super(DigitImageDatasetAugmentator, self).__init__(labels, images)
        """
        Construct a digit dataset from list of examples and labels and augments the data

        Parameters
        ----------

        labels:
        a np.array of type uint8 storing each image digit example label

        images:
        digit examples in a numpy array with format (N, height, width), where N is the number of examples


        labels length and images.shape[0] must have same size

        percent_augmentation:
        the percentage, related to original dataset size, of new examples to generate

        augmentation_method:
        the augmentation method to use.
        - elastic: augment digits by deforming it's shape by using an non-rigid transformation
        - affine: translates and rotates each digit randomly
        """

        self._percent_augmentation = percent_augmentation
        self._augmentation_method = augmentation_method

        # create a dictionary to store classes examples
        self._digit_examples = {}
        for i in range(10):
            mask = labels == i
            self._digit_examples[i] = images[mask]

        # store sample image shape as (width, height)
        self._sample_shape = images.shape[1:][::-1]

        self._generate_augmented_examples()


    def augment_from_digit_dataset(self, digit_dataset: DigitImageDataset) -> DigitImageDataset:
        """
        Load and generate dataset augmentation from an existing digit_dataset
        """
        pass

    def _generate_augmented_examples(self):

        # basic idea: go to each class and generate self._percent_augmentation new examples using elastic method
        pass

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
        return np.zeros((height, width), dtype=np.uint8)

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

            # random sample a digit from digit dataset
            digit_image = self._dataset.random_sample_digit(digit)

            # get relevant content bounding box
            x0, _, x1, _ = helpers.calculate_binary_image_contents_bbox(digit_image)

            # crop digit contents on x axis
            # adding 1 extra pixel in x1 (x max coordinate) because it marks relevant data pixel and we need to crop background only.
            digit_image = digit_image[:,x0:x1+1]

            intermediate_result_images.append(digit_image)

            # add spacing if not last element
            if index == N-1:
                continue

            random_spacing_block = self._generate_random_space_block()
            intermediate_result_images.append(random_spacing_block)

        result = np.hstack(intermediate_result_images)

        # finally, rescale to desired output width
        rescaled = helpers.rescale_to_width(result, self._output_image_width)

        # invert colors to make white background
        return 255 - rescaled

def get_or_load_digits_dataset():

    if this._DIGITS_DATASET is None:
        this._DIGITS_DATASET = DigitImageDataset()
        this._DIGITS_DATASET.load(_DIGITS_DATASET_FILEPATH)

    return this._DIGITS_DATASET

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

    # load digits dataset
    digits_dataset = get_or_load_digits_dataset()

    # Generate the synthetic sequence using the digit dataset created
    digit_sequence_generator = DigitSequenceImageGenerator(digits_dataset, spacing_range=spacing_range, output_image_width=image_width)

    return digit_sequence_generator.generate_from_data(digits)


