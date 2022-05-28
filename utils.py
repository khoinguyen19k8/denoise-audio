from cgi import test
import numpy as np 
import random
import tensorflow as tf

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
    train_dataset: List containing two numpy arrays, they have the shape of (train_size, 5500) and (train_size, 11000), respectively
    test_dataset: List containing two numpy arrays, they have the shape of (test_size, 5500) and (test_size, 11000), respectively
    test_ids: ids used to divide into train and test set from the original data, List[Int]
    """
    dataset_size = len(dataset[0])
    random.seed(seed)
    test_size = int((1 - ratio) * dataset_size)
    
    train_dataset = [[], []]
    test_dataset = [[], []]
    test_ids = set(random.sample(range(dataset_size), test_size))

    for i, (original_audio, noisy_audio) in enumerate(zip(dataset[0], dataset[1])):
        if i in test_ids:
            test_dataset[1].append(original_audio)
            test_dataset[0].append(noisy_audio)
        else:
            train_dataset[1].append(original_audio)
            train_dataset[0].append(noisy_audio)
    train_dataset = [np.array(item, dtype=np.float32) for item in train_dataset]
    test_dataset = [np.array(item, dtype = np.float32) for item in test_dataset]
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

def evaluate_tf(clean, denoised, snr_only = True):
    
    clean, denoised = tf.squeeze(clean), tf.squeeze(denoised)
    se = tf.math.reduce_mean((denoised - clean) ** 2, axis = 1)
    mse = tf.math.reduce_mean(se)
    mae = tf.math.abs(tf.math.reduce_mean(tf.math.reduce_mean(denoised - clean, axis = 1)))

    #SNR and PSNR
    num = tf.math.reduce_sum((clean**2), axis = 1)
    den = tf.math.reduce_sum(((denoised - clean) ** 2), axis = 1)
    ep = 1e-9
    SNR = 20* tf.math.reduce_mean(tf.experimental.numpy.log10(tf.math.sqrt(num)/(tf.math.sqrt(den) + ep)))

    if snr_only == True:
        return SNR
    else:
        return mse, mae, SNR 
    
def snr_tf(clean, denoised):
    clean, denoised = tf.squeeze(clean), tf.squeeze(denoised)
    
    #SNR and PSNR
    num = tf.math.reduce_sum((clean**2), axis = 1)
    den = tf.math.reduce_sum(((denoised - clean) ** 2), axis = 1)
    ep = 1e-9
    SNR = 20* tf.math.reduce_mean(tf.experimental.numpy.log10(tf.math.sqrt(num)/(tf.math.sqrt(den) + ep)))
    
    return SNR