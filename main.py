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

# softmax function: converts a vector of real numbers to a probability distribution

torch.random.manual_seed(seed=1234)

# data
text = "Hi!, My name is Jessi."
tokens = [13347, 0, 3092, 836, 374, 7011, 383, 355, 13]

#parameters
# zero index
vocab_size = max(tokens) + 1
emb_dim = 5
context = len(tokens)


#layers
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

query = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
key = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
value = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)

# mask filter
ones = torch.ones(size=[context, context], dtype=torch.float)
mask = torch.tril(input=ones)

# forward pass
# [9] -> [1, 9]
t_tokens = torch.tensor(data=tokens).unsqueeze(dim=0)
x = embedding(t_tokens)

B, T, C = x.size()
Q = query(x)
K = key(x)
V = value(x)

QK = Q @ K.transpose(-2, -1) * C**-0.5

# applying mask
attention = QK.masked_fill(mask[:T, :T] == 0, float('-inf'))

# [1,9,9] normalizing to 0 and 1 in embedding dimension
attention = F.softmax(input=attention, dim = -1)

# [1,9,9] @ [1, 9, 50] -> [1, 9, 50]
out = attention @ V

# data representation
print(out.size())
