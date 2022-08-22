# This module contains the class that deals with the processing of texts and the conversion into a numeric vector.
# A Vocabulary type object is required for its operation. Embedding matrices are not stored here.

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import *
import nltk
import numpy as np
import re
import torch


#nltk.download('stopwords')
#nltk.download('wordnet')
nltk.data.path.append("./model/nltk_data")


class TextProcessor(object):

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.stop = stopwords.words()
        self.lemmatizer = WordNetLemmatizer()


    # Clean a text and return a list of tokens
    def text_to_tokens(self, text: str) -> List[str]:
        # to lower case + regex to remove punctuations (and similar) + lemmatization + split into list
        text = text.lower()
        text = re.sub(r'(.)\1+', r'\1\1', text)
        text = re.sub("[^a-z0-9]+", " ", text)
        tokens = [self.lemmatizer.lemmatize(w) for w in text.split(" ") if w not in self.stop]
        return tokens


    # Convert a list of token(string) into a list of indices(int)
    def tokens_to_vector(self, tokens: List[str], max_len: int=32, pad: bool=True) -> Tuple[torch.Tensor,int]:

        # token to indices using vocabulary
        indices = [self.vocabulary.get_index(token) for token in tokens]

        # senteces is too longer, so we are cutting it to max_len
        if len(indices) > max_len:
            return  np.array(indices[:max_len],dtype=np.int64), max_len

        # sentence is too short, so we have to pad until max_len
        if pad:
            vector = np.zeros(max_len, dtype=np.int64)
            vector[:len(indices)] = indices
            vector[len(indices):] = self.vocabulary.pad_index
            return vector, len(indices)

        return indices,len(indices)


    # Concatenate the 2 vector of the sentences adding <SEP> token between
    def vectors_to_vector(self, vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        final_len = len(vector1) + 1 + len(vector2)
        final_vector = np.zeros(final_len, dtype=np.int64)

        final_vector[:len(vector1)] = vector1
        final_vector[len(vector1)] = self.vocabulary.sep_index
        final_vector[len(vector1)+1:] = vector2

        return final_vector


    # Takes a batch of text and returns a batch of vector of indices + the lengths
    def batch_to_input(self, batch_text1: List[str], batch_text2: List[str],
                       device:str ="cpu", merge: bool=False) -> Tuple[torch.Tensor,torch.Tensor]:

        input_matrix = []
        lengths=[]

        for text1,text2 in zip (batch_text1, batch_text2):

            tokens1 = self.text_to_tokens(text1)
            tokens2 = self.text_to_tokens(text2)
            vector1,len1  = self.tokens_to_vector (tokens1)
            vector2,len2  = self.tokens_to_vector (tokens2)

            if merge:
                input_matrix += [self.vectors_to_vector(vector1,vector2)]
                lengths += [len(vector1)+len(vector2)+1]
            else:
                input_matrix += [[vector1,vector2]]
                lengths += [[len1,len2]]

        return torch.tensor(input_matrix).to(device), torch.tensor(lengths, dtype=torch.int64)



