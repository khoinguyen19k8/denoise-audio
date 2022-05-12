from cgi import test
import numpy as np 
import random
from torch.utils.data import Dataset 
import torch

def train_test_split(dataset: Dataset, ratio = 0.8, seed = 42):
    """
    Split a dataset based on ratio into train and test set
    --------------------
    Parameters:
    dataset: a Dataset class subclassed from torch.utils.data.Dataset
    ratio: ratio dividing train and test set
    seed: seed for random generation
    --------------------
    Returns:
    train_dataset: List containing training data, List[List[np.array], List[np.array]]
    test_dataset: List containing testing data, List[List[np.array], List[np.array]]
    test_ids: ids used to divide into train and test set from the original data, List[Int]
    """
    dataset_size = len(dataset)
    random.seed(seed)
    test_size = int((1 - ratio) * dataset_size)
    
    train_dataset = [[], []]
    test_dataset = [[], []]
    test_ids = set(random.sample(range(dataset_size), test_size))

    for i in range(dataset_size):
        original_audio, noisy_audio = dataset[i]
        if i in test_ids:
            test_dataset[0].append(original_audio)
            test_dataset[1].append(noisy_audio)
        else:
            train_dataset[0].append(original_audio)
            train_dataset[1].append(noisy_audio)
    return train_dataset, test_dataset, test_ids

def snr(signal, noise):
    """
    This function calculates Signal-to-Noise ratio (SNR), an algorithm which compares the level
    of a desired signal to the level of background noise. A ratio higher than 1:1 indicates more signal than noise
    When training the model we would use MSE/MAE to gauge performance of models. 
    This function is used to gauge the quality of audio denoised by our models.
    --------------------
    Parameters:
    signal: pytorch Tensor of shape (11000,) 
    noise: pytorch Tensor of shape (5500,)
    --------------------
    Returns:
    Signal-to-noise ratio
    """
    signal_length, noise_length = signal.shape[0], noise.shape[0]
    signal_mean_squares = (1 / signal_length) * torch.sum(torch.square(signal))
    noise_mean_squares = (1 / noise_length) * torch.sum(torch.square(noise))
    return signal_mean_squares / noise_mean_squares
    
