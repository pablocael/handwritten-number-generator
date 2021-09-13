import numpy as np
from number_generator import core
from number_generator import generate_numbers_sequence

# configure the dataset to be local
core._DIGITS_DATASET_FILEPATH = 'data/digits-dataset.pickle'

def test_empty_sequence():

    result = generate_numbers_sequence([], (5,10), 100)

    # width must be correct even for empyt sequences
    assert result.shape[1] == 100

    # empty sequence must be all background
    assert (result == 255).all()

def test_output_size():

    # generate random sequences of different lengths and check if output width is correct
    width = 210

    num_sequences = 50

    digits = [i for i in range(10)]

    for i in range(num_sequences):

        min_spacing = np.random.randint(1,5)
        max_spacing = np.random.randint(5,15)

        sequence_size = np.random.randint(1, 100)
        sequence = np.random.choice(digits, size=sequence_size, replace=True)

        result = generate_numbers_sequence(sequence, spacing_range=(min_spacing, max_spacing), image_width=width)

        # all generated sequences must have precise correct output size
        assert result.shape[1] == width

        # resulting image cannot contain background only (black image)
        assert not (result == 255).all()



