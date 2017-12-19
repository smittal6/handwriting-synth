import math
import os
import sys
# sys.path.insert(0,'..')
import numpy as np
import torch
from torch.utils.data import Dataset


class StrokesDataset(Dataset):
    """Strokes dataset"""

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.strokes = np.load('data/strokes.npy')
        with open('data/sentences.txt') as f:
                self.texts = f.readlines()
        self.transform = transform

    def __len__(self):
        return self.strokes.shape[0]

    def __getitem__(self, idx):
        init = self.strokes[idx]
        text = self.texts[idx].rstrip()
        print text
        one_hot_code = self.getOneHot(text)
        next_stroke = np.zeros(init.shape)
        next_stroke[:-1,:] = init[1:,:]
        sample = {'initial': init, 'next': next_stroke, 'text': text, 'encoding': one_hot_code}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def getOneHot(self,string):
        """
        Returns the one hot representation of the characters of the string
        """
        alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".-'
        vec_len = len(alphabet)
        encoding = []
        for s in string:
            temp = np.zeros(vec_len)
            try:
                temp[alphabet.index(s)] = 1
            except ValueError:
                continue
            encoding.append(temp)
        final_encoding = np.asarray(encoding)
        return final_encoding
