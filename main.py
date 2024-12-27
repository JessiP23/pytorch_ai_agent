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

# torch.random.manual_seed(seed=1234)

# data
# text = "Hi!, My name is Jessi."
# tokens = [13347, 0, 3092, 836, 374, 7011, 383, 355, 13]

#parameters
# zero index
# vocab_size = max(tokens) + 1
# emb_dim = 5
# context = len(tokens)


#layers

# positional encoding
# pe = nn.Embedding(num_embeddings=context, embedding_dim=emb_dim)

# embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

# query = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
# key = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
# value = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)

# mask filter
# ones = torch.ones(size=[context, context], dtype=torch.float)
# mask = torch.tril(input=ones)

# introducing indices
# indices = torch.arange(context, dtype=torch.long)

# forward pass
# [9] -> [1, 9]
# t_tokens = torch.tensor(data=tokens).unsqueeze(dim=0)
# x = embedding(t_tokens)

# [1, 9, 50] + [1, 9, 50] -> [1, 9, 50]
# x = pe(indices) + x

# B, T, C = x.size()
# Q = query(x)
# K = key(x)
# V = value(x)

# QK = Q @ K.transpose(-2, -1) * C**-0.5

# applying mask
# attention = QK.masked_fill(mask[:T, :T] == 0, float('-inf'))

# [1,9,9] normalizing to 0 and 1 in embedding dimension
# attention = F.softmax(input=attention, dim = -1)

# [1,9,9] @ [1, 9, 50] -> [1, 9, 50]
# out = attention @ V

# data representation
# print(out.size())

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pe = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        word_emb = self.embedding(x)
        word_pe = self.pe(x)
        return word_emb + word_pe

class AttentionBlock(nn.Module):
    
    def __init__(self, embedding_dim, context_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        ones = torch.ones(size=[context_size, context_size], dtype=torch.float)
        
        # Triangular matrix
        self.register_buffer(name="mask", tensor=torch.tril(input=ones))
        
    def forward(self, x):
        B, T, C = x.size()
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        qk = query @ key.transpose(-2, -1) * C**-0.5
        attention = qk.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        
        out = attention @ value
        return out


# the vector representation of each token must be different if we change the order of the tokens
# Positional Encoding: Encode each token position with a fixed value for each position.

class MultiAttentionBlock(nn.Module):
    
    def __init__(self, embedding_dim, num_heads, context_size):
        """

        Args:
            embedding_dim (init): Dimensions of the embedding
            num_heads (int): Number of attention Heads
            context_size (int): Size of the context window.
        """
        super().__init__()
        
        # Determine the number of heads
        head_dimension = embedding_dim // num_heads
        
        assert head_dimension * num_heads == embedding_dim, "Embedding dimension must be divisible by the number of heads."
        
        self.attention = nn.ModuleList(modules=[AttentionBlock(embedding_dim, head_dimension, context_size) for _ in range(num_heads)])
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        
    def forward(self, x):
        """
        MultipAttentionBlock forward pass layer

        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
        torch.Tensor
        """
        
        out = torch.cat(tensors=[attention(x) for attention in self.attention], dim = -1)
        
        x = self.linear(out)
        
        return x

# Apply skip-connect after the Multi-Head-Attention block to avoid vanishing gradients
# Apply a Batch normalization and feed a Fully-Connected-Neurons to process the useful information extracted from previous block
# Feed teh previous block to another Attention block


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        """
        Foward pass of the feed Forward Layer

        Args:
            x (torch.Tensor): Input tensor
        returns:
            torch.Tensor: output tensor
        """
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_dim, context_size, hidden_dim):
        """
        Initialize the Decoder Layer

        Args:
            embedding_dim (int): Word embedding dimension
            head_dim (int): Head dimension
            context_size (int): Size of the context window
            hidden_dim (int): Feed Forward hidden dimension layer
        """
        super().__init__()
        
        self.attention = MultiAttentionBlock(embedding_dim, head_dim, context_size)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        self.norm_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embedding_dim)
    
    def forward(self, x):
        """
        Forward pass of teh decoder layer

        Args:
            x (torch.Tensor): input tensor
        """
        
        x_norm = self.norm_1(x)
        attention = self.attention(x_norm)
        attention = attention + x
        
        attention_norm = self.norm_2(attention)
        feed_forward = self.feed_forward(attention_norm)
        feed_forward = feed_forward + attention
        
        return feed_forward