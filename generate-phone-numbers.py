import os
import sys
import argparse
from number_generator import generate_numbers_sequence

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
arggroup.add_argument('--output-path', dest='output_path', type=str,
                    help='The path where to store the genereate dataset images', required=True)

try:
    args = parser.parse_args()
    min_spacing = args.min_spacing
    max_spacing = args.max_spacing
    image_width = args.image_width
    output_path = args.output_path

except ValueError as ve:
    print(ve)
except:
    sys.exit(0)

if __name__ == '__main__':

    filename = os.path.join(output_path, ''.join(sequence))
    print('filename', filename)

