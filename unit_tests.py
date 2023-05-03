import os, gc, sys, time, datetime, math, random, copy, psutil
import numpy as np,  matplotlib.pyplot as plt, pandas as pd
from   tqdm.auto import tqdm
import torch, torch.nn as nn

from qunet import  Config, Data, MLP, CNN, ResCNN, SelfAttention, FFT, TransformerBlock, Transformer, PointsBlock

print("--------------------------")

res = True
res = res and SelfAttention.unit_test()
res = res and FFT.unit_test()
res = res and TransformerBlock.unit_test()
res = res and Transformer.unit_test()
res = res and CNN.unit_test()
res = res and ResCNN.unit_test()
res = res and Data.unit_test()

print("--------------------------")
print("unit tests result = ", res)


from torchinfo import summary  
#cnn = ResCNN(input=(3, None,None), channel=[16, 32],  pool_ker=[2,2], batchnorm=1, drop=0.1)
cnn = CNN(input=(3, None,None), channel=[16, 32], pool_ker=[2,2], batchnorm=1, drop=0.1)
summary(cnn, input_data = torch.rand(1,3, 64,32), col_names=["input_size", "output_size", "num_params"], col_width=18, depth=4) 
x = torch.rand(1,3, 16, 16)
y = cnn(x)
print(y.shape)