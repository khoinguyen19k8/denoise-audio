from torch.utils.data import Dataset
import numpy as np

class DenoiseDataset(Dataset):
    """
    Dataset class to load the dataset
    -------
    Paramaters:
    self.original_audio: numpy array that contains original audio recordings, this array has the shape (num_recordings, 11000)
    self.noisy_audio: numpy array that contains noisy and downsampled audio recordings, this array has the shape (num_recordings, 5500)
    """
    def __init__(self, noisy_audio: np.array, original_audio: np.array, original_transform = None, noisy_transform = None):
       self.original_audio = original_audio
       self.noisy_audio = noisy_audio 
       self.original_transform = original_transform
       self.noisy_transform = noisy_transform
    
    def __len__(self):
        return len(self.original_audio.shape[0])

    def __getitem__(self, index: int):
        org_audio = self.original_audio[index]
        noisy_audio = self.noisy_audio[index]

        if self.original_transform:
            org_audio = self.original_transform(org_audio)
        if self.noisy_transform:
            noisy_audio = self.noisy_transform(noisy_audio)
        return noisy_audio, org_audio 
