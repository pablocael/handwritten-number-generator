
# number-generator 0.0.1

Introduction
----------------------

*number-generator* is package for synthesizing images of handwritten sequences of digits, with digits randomly sampled from real handwritten digits dataset.

Automatically generated handwritten digit sequences can be used for training OCR models without the need of manual work.

The method for generating handwritten digit sequence adopted in this package is based on using a pre-defined digits dataset (called default digits dataset) to compose the sequences. For composing new sequences, digits are randomly sampled from the default digit dataset.

This package provides two utility scripts:

- *generate-numbers-sequence*: generates individual number sequences with customized spacing range and output width

- *generate-phone-numbers*: generates phone numbers-like number sequence datasets with custom number of examples

By default, the default digit dataset will be used by the above scripts without any configuration needed. The default digit dataset its based on the [MNIST dataset](https://data.deepai.org/mnist.zip). This default digit dataset, however, can be changed by user, as shown in section 2.2.

This package also provides serveral utility classes within ```number_generator.core``` module. The utility classes are:

- core.DigitImageDataset:
	Allows easily loading / saving digits dataset as well as sampling digits from digit classes.
	
- core.DigitSequenceImageGenerator:
	Allows building images of sequence of images based on an input dataset.
	
A convenience API for generating number sequence images is provided in the main module. How to use this api is shown in the example below:

```py
from number_generator import generate_numbers_sequence

result_image = generate_number_sequence([1,2,3], spacing_range=(4,10), image_width=100)
```


*This package is built for python >= 3.6*


Installation
----------------------
```bash
pip3 install number-generator==0.0.1
```


Dependencies
----------------------

This package was build with a minimalist paradigm, trying to avoid bloatware package dependency.

Dependencis are the following:

- numpy >= 1.21.2

	Necessary to perform array processing and mathematical operations
 	This package requires 93M of space.
 
- pytest >= 6.2.5
	
	Necessary to perform unit tests
	This package requires 1.9M of space.
	
- argparse >= 1.4.0

	Argparse is used to handle input arguments in a more high level, scalable and organized fashion.
	This package requires just a few KB of free space.
	
- Pillow >= 8.3.2

	Necessary to save images in proper format.
	This package requires 11M of space.


Usage
----------------------

#### 1. Generating a handwritten number a sequence from command line:

Example:

```py
generate-numbers-sequence --sequence=12345678 --min-spacing=5 --max-spacing=8 --image-width=200
```

Output:

```console
---------------------------
Digits sequence generator
---------------------------
Synthesizing image for digits sequence "12345678" ...
Image of dimensions 200 x 28 generated successfully at "./12345678.png", proccess took 0.1 seconds...
```

the output image is saved in the current directory by default, but can be saved in any other directory by specifying the option ```--output-path```

#### 2. Generating a phone number sequences dataset from command line:

Example:

```py
generate-phone-numbers.py --num-images=200 --min-spacing=5 --max-spacing=10 --image-width=100 --output-path=./
```

Output:

```console
---------------------------------
Phone numbers dataset generator
---------------------------------
Creating phone numbers like database with 200 examples, this might take a few moments ...
Dataset with 200 telephone like images generated successfully at "./phonenumbers-handwritten-dataset-200.pickle", proccess took 0.3 seconds...
```

### 2.1 Using the generated phone number sequences dataset:

The generated dataset is serialized with pickle in the given format:

```py
{
	'images': np.ndarray,
	'labels': List[str],
	'metadata': {
            'creation_timestamp': float # timestamp of the time of generation
            'spacing_range': (min_spacing, max_spacing),
            'image_width': image_width
        }
}
```

To load the dataset using pickle:

```py
import pickle
data = None
with open(input_filepath, 'rb') as handle:
	data = pickle.load(handle)
```

Alternativelly (and conveniently) its possible to use the class ```core.GenericDataset``` to both load and save datasets:

```py
from number_generator.core import GenericDataset
dataset = GenericDatset()
dataset.load('datasetfilepath.pickle')

# access first image and label
img, label = dataset[0]

# read metadata
metadata = dataset.get_metadata()
print(metadata['creation_timestamp'])
```

### 2.2 Using custom digit datasets:

By default, number-generator package ships a native dataset based on MNIST. However, its possible change the default dataset to a user custom dataset, as long as the dataset is still composed by square images (width equals height). To change default database, set 'NG\_DEFAULT\_DIGIT\_DATASET' environment variable to the path of the database. The custom database must be in the same format shown in item 2.1.
See number\_generator.core.DigitImageDataset for more information.

Example on how to change default input digit dataset:

```console
export NG_DEFAULT_DIGIT_DATASET='./mypath/my_digits_dataset.pickle'
```

One can also change default dataset programatically as shown below:

```py
from number_generator import set_default_dataset_filepath

set_default_dataset_filepath('./mynewpath/mydataset.pickle')
```

This will force current default dataset to change.

Help
----------------------

Support can be provided through email: pablo.cael@gmail.com.

Executable scripts have help info can be access by using the ```--help``` option:

```bash
generate-numbers-sequence --help
```

```console
usage: generate-numbers-sequence [-h] --sequence SEQUENCE --min-spacing MIN_SPACING --max-spacing MAX_SPACING --image-width IMAGE_WIDTH [--output-path OUTPUT_PATH]

Generate a dataset of images containing random sequences in the format of Japanese phone numbers

required arguments:
  --sequence SEQUENCE   The minimum spacing to use between consecutive digits
  --min-spacing MIN_SPACING
                        The minimum spacing to use between consecutive digits
  --max-spacing MAX_SPACING
                        The maximum spacing to use between consecutive digits
  --image-width IMAGE_WIDTH
                        The output image width of each generated example in the generated dataset

optional arguments:
  --output-path OUTPUT_PATH
                        The path where to store the generated dataset images
```
Development
----------------------

###  Testing:
number-generator uses pytest. To run the tests, run:

```py
python3 -m pytest
```

on the root directory.


Future Improvement and Features
----------------------

1. Genereate phone numbers with dashes between number blocks
2. Generate number sequences using similar handwritten characters (to simulate same person writting the sequence): this can be achieved by clustering each digit class using k-means and randomly choosing a fixed cluster for each class (or maybe more than one cluster, based on similarity)

