## this file contains the basic structure of the transformer
## basically just the classes for the encoder and decoder blocks + the linear and softmax at the end

import torch.nn as nn
import copy


## helper functions
def clone(module, N):
    '''
    takes a layer and clones it into N identical layers
    if you are unfamiliar with deepcopies, basically they create independant copies with no shared references
    this is very useful for our case as we dont want different layers affecting each other unless explicitly so
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

