"""
Generate an image of handwritten digit sequences from an input string containing digit characters 0-9.
"""

import os
import sys
import time
from PIL import Image
import argparse
from number_generator import generate_numbers_sequence

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a dataset of images containing random sequences in \
                                        the format of Japanese phone numbers')
    parser._action_groups.pop()
    arggroup = parser.add_argument_group('required arguments')

    arggroup.add_argument('--sequence', dest='sequence', type=str,
                        help='The minimum spacing to use between consecutive digits', required=True)
    arggroup.add_argument('--min-spacing', dest='min_spacing', type=int,
                        help='The minimum spacing to use between consecutive digits', required=True)
    arggroup.add_argument('--max-spacing', dest='max_spacing', type=int,
                        help='The maximum spacing to use between consecutive digits', required=True)
    arggroup.add_argument('--image-width', dest='image_width', type=int,
                        help='The output image width of each generated example in the generated dataset', required=True)

    optgroup = parser.add_argument_group('optional arguments')
    optgroup.add_argument('--output-path', dest='output_path', type=str, default='./',
                        help='The path where to store the genereate dataset images')

    def is_sequence_valid(seq):
        digits = [str(i) for i in range(10)]
        for c in seq:
            if c not in digits:
                return False

        return True

    # initialize argument variables
    min_spacing = None
    max_spacing = None
    image_width = None
    output_path = None
    sequence = None

    try:
        args = parser.parse_args()
        min_spacing = args.min_spacing
        max_spacing = args.max_spacing
        image_width = args.image_width
        output_path = args.output_path
        sequence = args.sequence
        if not is_sequence_valid(sequence):
            raise ValueError('error, --sequence argument must be a string containing only digits from 0 to 9')

    except ValueError as ve:
        print(ve)
        sys.exit(0)

    filename = os.path.join(output_path, ''.join(sequence))
    filename = f'{filename}.png'

    digits = [int(c) for c in sequence]

    print()
    print(f'---------------------------')
    print(f'Digits sequence generator')
    print(f'---------------------------')

    print(f'Synthesizing image for digits sequence "{sequence}" ...')

    try:
        start_time = time.time()
        output_image = generate_numbers_sequence(digits=digits, spacing_range=(min_spacing, max_spacing), image_width=image_width)

        im = Image.fromarray(output_image)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(filename)
        end_time = time.time()
        print(f'Image of dimensions {im.width} x {im.height} generated successfully at "{filename}", proccess took {end_time-start_time:.1f} seconds...')

    except Exception as e:
        print('an error occurred while trying to synthesize digit image:', e)
        print('please contact pablo.cael@gmail.com for support')

    print()
