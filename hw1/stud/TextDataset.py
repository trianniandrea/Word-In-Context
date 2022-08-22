# This module contains the class that manages the dataset. It inherits from torch.utils.data.dataset.
# Can handle training and dev set, it also takes care of loading jsonl data from disk.
# It Implements the iterator pattern, has its own length, and does some very light processing to samples.


import json
import random
from typing import *
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, train_data: List[Dict], val_data:List[Dict]):

        self.train_data = train_data
        self.val_data = val_data
        self.__lookup_dict = {'train': self.train_data, 'val': self.val_data}
        self.__target_split = "train"


    # Swap between train and test set.
    def set_split(self, split: str ="train") -> None:
        self.__target_split = split



    # Load the dataset given the filepaths, it shuffle the data before call the constructor.
    @classmethod
    def from_disk(cls, train_path: str, val_path: str):

        train_data = []
        with open(train_path) as file:
            for sample in list(file):
                train_data += [json.loads(sample)]
        random.shuffle(train_data)

        val_data = []
        with open(val_path) as file:
            for sample in list(file):
                val_data += [json.loads(sample)]
        random.shuffle(val_data)

        return cls(train_data,val_data)


    # Returns the length of the actual set.
    def __len__(self) -> int:
        return len(self.__lookup_dict[self.__target_split])


    # Returns a sample from the actual set.
    def __getitem__(self, index: int) -> dict:
        sample = self.__lookup_dict[self.__target_split][index]
        return {'x1_data': sample['sentence1'],'x2_data':sample['sentence2'],'y_target': 1.0 if sample['label']=="True" else 0.0,
                'lemma':sample['lemma'], 'pos':sample['pos']}

