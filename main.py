import torch

# help create train neural networks
import torch.nn as nn
import torch.nn.functional as F

# neural net torch.nn
# @ stands for matrix multiplication

# tokenization - TikToken by OpenAI or SentencePiece by Google

# SentencePiece mainly for Neural Network-based text generation systems where teh vocab is determined prior to neural model training.

# Tiktoken is used with OpenAI's models

class Tokenizer:
    
    @staticmethod
    
    def create_vocab(dataset):
        
        
        """
        Create a vocab from a dataset

        Args:
        dataset (txt)
        
        Output:
        Dict[str, int]
        """
        
        # enumerate returns index and token
        # example : ['apple', 'banana', 'apple'] -> { 0: 'apple', 1 : 'banana'}
        vocab = {
            
            # token: index for index is a dictionary comprehension
            token: index for index, token in enumerate(sorted(list(set(dataset))))
        }
        
        # Adding unknown token
        vocab["<UNK>"] = len(vocab)
        
        return vocab
    
    
    # encode: string to integer
    # decode: integer to string
    
    # initialize the instance of a class
    # vocab is a dictionary
    def __init__(self, vocab):
        
        # vocab.items returns a list of the dictionaty's key-value pairs
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}
        
        # Reverse the vocab
        # each value v becomes a key and each key k becomes a value
        self.vocab_decode = {v: k for k, v in self.vocab_encode.items()}
        
    def encode(self, text):
        # arguments: text to be encoded
        # returns: list of token indices
        return [self.vocab_encode.get(char, self.vocab_encode["<UNK>"]) for char in text]
    
    def decode(self, indices):
        # arguments: list of token indices
        # returns: text
        return "".join([self.vocab_decode.get(index, "<UNK>") for index in indices])
    

# Add meaning to static numbers
# Embeddings: Sense to numerical values
# Embedding Layer: Converts token indices to dense vectors of fixed size

# 1, 2, 3,4 , 5, 6, 7 ... , 1000 -> vocab size
# mouse is 12
# access 12 and 12 has the vector representation of mouse

# cosine distance -> similarity between two vectors