## this file contains the basic structure of the transformer
## basically just the classes for the encoder and decoder blocks + the linear and softmax at the end

import torch
import torch.nn as nn
import copy


## helper functions and classes
def clone(module, N):
    '''
    takes a layer and clones it into N identical layers
    if you are unfamiliar with deepcopies, basically they create independant copies with no shared references
    this is very useful for our case as we dont want different layers affecting each other unless explicitly so
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.module):
    '''
    as the name suggests, performs layer normalization
    very important as it makes operations far less computationally expensive
    basically batch norm but across a layer instead of a batch
    (see https://arxiv.org/abs/1607.06450 for more details)
    '''
    
    def __init__(self, features, eps=1e-6):
        '''
        features here is the number of features in the input layer (size of input tensor's final dimension)
        epsilon is a very small value that is here so we dont try to divide by 0 or anything like that

        initializes a_2 and b_2 as a matrix of ones and a matrix of zeroes of size features respectively
        '''
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        '''
        this is the forward prop function

        calculates the mean of the input tensor along the last dimension (retains input dimensionality)
        calculates standard deviation of the input tensor, also along the last dimension (retains input dimensionality)

        normalizes the input by subtracting the mean, then divided by the standard deviation (plus epsilon so no /0 nonsense)
        this makes the mean and unit variance both 0 (meaning the mean is now 0 and the variance is now 1 (meaning standard deviation is now also 1))

        this normalized input is then scaled by a_2 and shifted by b_2
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * ((x - mean) / (std + self.eps)) + self.b_2


## architecture blocks
class Encoder():
    '''
    encoder block of the transformer architecture (left side of paper's diagram)
    '''
    

class Decoder():
    '''
    decoder block of the encoder architecture (right side of paper's diagram)
    '''
    pass

