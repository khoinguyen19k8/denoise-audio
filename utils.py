from cgi import test
import numpy as np 
import random
from torch.utils.data import Dataset 
import torch

def train_test_split(dataset, ratio = 0.8, seed = 42):
    """
    Split a dataset based on ratio into train and test set
    --------------------
    Parameters:
    dataset: a list contains two lists, one for original audio and the other for noisy audio.
    ratio: ratio dividing train and test set
    seed: seed for random generation
    --------------------
    Returns:
    train_dataset: List containing training data, List[List[np.array], List[np.array]]. The two lists are noisified and original audio, respectively.
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
        noisy_audio, original_audio= dataset[i]
        if i in test_ids:
            test_dataset[1].append(original_audio)
            test_dataset[0].append(noisy_audio)
        else:
            train_dataset[1].append(original_audio)
            train_dataset[0].append(noisy_audio)
    return train_dataset, test_dataset, test_ids

def evaluate(clean, denoised):
    """"
    This function compares two set of signals by calculating the MSE (Mean squared error), MAE (Mean absolute error),
    and SNR (signal to noise ratio) in db averaged over all the signals.
    Receives two matrices of shape N, D. That correspond to N signals of length D.
    clean: a 2D numpy array containing the clean (original) signals.
    denoised: a 2D numpy array containing the denoised (reconstructed) versions of the original signals.
    """

    #MSE and MAE
    se = ((denoised - clean) ** 2).mean(-1)
    mse = se.mean()
    mae = np.abs(denoised - clean).mean(-1).mean()

    #SNR and PSNR
    num = (clean**2).sum(-1)
    den = ((denoised - clean) ** 2).sum(-1)
    ep = 1e-9
    SNR = 20*np.log10(np.sqrt(num)/(np.sqrt(den) + ep)).mean()

    return mse, mae, SNR 
