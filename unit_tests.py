import os, gc, sys, time, datetime, math, random, copy, psutil
import numpy as np,  matplotlib.pyplot as plt, pandas as pd
from   tqdm.auto import tqdm
import torch, torch.nn as nn

from qunet import  Config, Data, MLP, CNN, ResCNN, SelfAttention, FFT, TransformerBlock, Transformer, PointsBlock

print("--------------------------")

res = True
res = res and MLP.unit_test()
res = res and CNN.unit_test()
res = res and ResCNN.unit_test()
res = res and SelfAttention.unit_test()
res = res and FFT.unit_test()
res = res and TransformerBlock.unit_test()
res = res and Transformer.unit_test()
res = res and Data.unit_test()

print("--------------------------")
print("unit tests result = ", res)