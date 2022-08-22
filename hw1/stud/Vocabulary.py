# This module contains the class that implements the vocabulary.
# The vocabulary is an abstraction that maps tokens (words) to unique integer indices.
# It can contain special tokens and can be saved and loaded from the hard drive.

import json


class Vocabulary():

    def __init__(self, tokens: dict = None, max_size: int = 10000,
                 unk: bool = False, pad: bool = False, sep:bool = False):

        self.tokens = tokens if tokens is not None else {}
        self.max_size = max_size if max_size > len(self) else len(self)
        self.unk_index = self.add_token("<UNK>") if unk else -1
        self.pad_index = self.add_token("<PAD>") if pad else -1
        self.sep_index = self.add_token("<SEP>") if sep else -1


    # Adds a token on top of the dictionary, if not present.
    # It returns the associated index.
    def add_token(self, token_ins: str) -> int:
        if token_ins not in self.tokens.keys():
            if len(self) < self.max_size:
                self.tokens[token_ins] = len(self)
            else:
                return -1
        return self.tokens[token_ins]


    # Given an index, return the associated token.
    def get_token(self, index_src: int) -> str:
        for token,index in self.tokens.items():
            if index == index_src:
                return token;
        return "";


    # Given a token, return its index.
    def get_index(self, token: str) -> int:
        return self.tokens.get(token, self.unk_index)


    # Retrieve an item (token) using an integer index (iterator pattern).
    def __getitem__(self, index: int) -> str:
        return self.get_token(index)


    # Return the length of the vocabulary.
    def __len__(self) -> int:
        return len(self.tokens)


    # Serialize the Vocabulary object into a dict (usefull to save to disk).
    def to_serializable(self) -> dict:
        return { 'tokens': self.tokens, 'max_size': self.max_size,
                 'unk_index': True if self.unk_index >= 0 else False,
                 'pad_index': True if self.pad_index >= 0 else False,
                 'sep_index': True if self.sep_index >= 0 else False }


    # Deserialize the content and return the original object (usefull to load from disk).
    @classmethod
    def from_serializable(cls, contents: dict):
        return cls(contents['tokens'], contents['max_size'],
                   contents['unk_index'], contents['pad_index'],contents['sep_index'])


    # Load a saved vocabulary from disk.
    @staticmethod
    def from_disk(filepath:str):
        with open(filepath) as fp:
            return Vocabulary.from_serializable(json.load(fp))


    # Save the vocabulary object to disk.
    def to_disk(self, filepath:str) -> None:
        with open(filepath, "w") as fp:
            json.dump(self.to_serializable(), fp)

