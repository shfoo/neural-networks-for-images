import numpy as np
import os
import pickle

def load_data(data_dir):
    """
    Input:
    - data_dir: Directory of MNIST data

    Output:
    - data: Dictionary containing training/ test data and labels
    """
    data = {'Xtrain': None, 'ytrain': None,
            'Xtest': None, 'ytest': None}

    data['Xtrain'] = load_images(data_dir+'train-images.idx3-ubyte')
    data['ytrain'] = load_labels(data_dir+'train-labels.idx1-ubyte')
    data['Xtest'] = load_images(data_dir+'t10k-images.idx3-ubyte')
    data['ytest'] = load_labels(data_dir+'t10k-labels.idx1-ubyte')

    return data

def load_images(data_path):
    """
    Input:
    - data_path: Path of MNIST image data (binary file)
    
    Output:
    - image_data: Image data (numpy array)
    """
    with open(data_path, 'rb') as f:
        magic_nbr = f.read(4)
        num_examples = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')

        image_data = np.uint8(np.zeros((num_examples, num_rows, num_cols)))

        for n in range(num_examples):
            for i in range(num_rows):
                for j in range(num_cols):
                    image_data[n, i, j] = int.from_bytes(f.read(1), byteorder='big')

    return image_data


def load_labels(data_path):
    """
    Input:
    - data_path: Path of MNIST label data (binary file)

    Output:
    - label_data: Label data (numpy array)
    """
    with open(data_path, 'rb') as f:
        magic_nbr = f.read(4)
        num_examples = int.from_bytes(f.read(4), byteorder='big')

        label_data = np.uint8(np.zeros((num_examples,)))

        for n in range(num_examples):
            label_data[n] = int.from_bytes(f.read(1), byteorder='big')

    return label_data


def load_and_pickle(data_dir):
    """
    Loads MNIST data and saves to disk as pickle file

    Input:
    - data_dir: Directory of MNIST data
    """
    data = load_data(data_dir)

    with open(data_dir+'mnist.pkl', 'bw') as p:
        pickle.dump(data, p)


def load_from_pickle(data_path):
    """
    Loads MNIST data from pickle file

    Input:
    - data_path: Path of data file in pickle file
    """
    with open(data_path, 'br') as p:
        data = pickle.load(p)

    return data
