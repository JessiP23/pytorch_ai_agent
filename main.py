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
    
    def __init__(self, vocab):
        
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}
        
        # Reverse the vocab
        self.vocab_decode = {v: k}