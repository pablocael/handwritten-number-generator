#!/usr/bin/env python3

"""
Generate a dataset of randomly generated handwritten telephone numbers using Japanese telephone number format.

Japanese telephone numbers vary between 3 3 4 and 3 4 4 digits format
"""
import os
import sys
import time
import argparse
import numpy as np

from number_generator.core import GenericDataset
from number_generator import generate_numbers_sequence

def generate_phone_like_sequence(size):

    sequence = [0] # phones always start with zero

    non_zero = [i for i in range(1,10)]
    all_digits = [0] + non_zero

    # choose a non zero number after first zero
    sequence.append(np.random.choice(non_zero, 1)[0])

    # choose the remaining digits
    sequence += np.random.choice(all_digits, size=size-2, replace=True).tolist()
    return sequence

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a dataset of images containing random sequences in \
                                        the format of Japanese phone numbers')
    parser._action_groups.pop()
    arggroup = parser.add_argument_group('required arguments')

    arggroup.add_argument('--min-spacing', dest='min_spacing', type=int,
                        help='The minimum spacing to use between consecutive digits', required=True)
    arggroup.add_argument('--max-spacing', dest='max_spacing', type=int,
                        help='The maximum spacing to use between consecutive digits', required=True)
    arggroup.add_argument('--image-width', dest='image_width', type=int,
                        help='The output image width of each generated example in the generated dataset', required=True)
    arggroup.add_argument('--num-images', dest='num_images', type=int,
                        help='The number of images to generate within the dataset', required=True)
    arggroup.add_argument('--output-path', dest='output_path', type=str,
                        help='The path where to store the genereate dataset images', required=True)

    min_spacing = None
    max_spacing = None
    image_width = None
    output_path = None
    num_images = None

    try:
        args = parser.parse_args()
        min_spacing = args.min_spacing
        max_spacing = args.max_spacing
        image_width = args.image_width
        output_path = args.output_path
        num_images = args.num_images

    except ValueError as ve:
        print(ve)
        sys.exit(0)

    images = []
    labels = []

    sequence_size = [
        10, # 334 format
        11, # 344 format
    ]

    print()
    print(f'---------------------------------')
    print(f'Phone numbers dataset generator')
    print(f'---------------------------------')

    print(f'Creating phone numbers like database with {num_images} examples, this might take a few moments ...')

    try:
        start_time = time.time()
        for i in range(num_images):
            # genereate phone number like sequence with random sequence size choosen from sequence_size
            sequence = generate_phone_like_sequence(np.random.choice(sequence_size, 1)[0])

            # append label as a string representing the digit sequence
            labels.append(''.join([str(d) for d in sequence]))

            generate_image = generate_numbers_sequence(sequence, spacing_range=(min_spacing, max_spacing), image_width=image_width)
            images.append(generate_image)

        dataset = GenericDataset(labels, images)

        output_filename = os.path.join(output_path, f'phonenumbers-handwritten-dataset-{num_images}.pickle')

        # add some metadata to be able to make sense of this dataset later
        metadata = {
            'creation_timestamp': time.time(),
            'spacing_range': (min_spacing, max_spacing),
            'image_width': image_width
        }

        dataset.save(output_filename, metadata)
        end_time = time.time()

        print(f'Dataset with {num_images} telephone like images generated successfully at "{output_filename}", proccess took {end_time-start_time:.1f} seconds...')

    except Exception as e:
        print('an error occurred while trying to generate the dataset:', e)
        print('please contact pablo.cael@gmail.com for support')


    print()




