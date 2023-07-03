import os, gc, sys, time, datetime, math, random, copy, psutil
import numpy as np,  matplotlib.pyplot as plt, pandas as pd
from   tqdm.auto import tqdm
import torch, torch.nn as nn

from qunet import  ModelState, Trainer, Config, Data, MLP, ResMLP, CNN, ResCNN, Attention, FFT, TransformerBlock, Transformer, VitEmb, Vit2d, PointsBlock

print(torch.__version__)
print(torch.cuda.is_available())
print("--------------------------")

res = True
res = res and MLP.unit_test()
res = res and ResMLP.unit_test()
res = res and CNN.unit_test()
res = res and ResCNN.unit_test()
res = res and Attention.unit_test()
res = res and FFT.unit_test()
res = res and TransformerBlock.unit_test()
res = res and Transformer.unit_test()
res = res and VitEmb.unit_test()
res = res and Vit2d.unit_test()
res = res and Data.unit_test()

print("--------------------------")
print("unit tests result = ", res)
print("--------------------------")

"""
cfg = Config(E=64, H=4, n_blocks=1, is_fft=0, is_att=1, res=2, gamma=0.2)
model = Transformer(cfg)
state = ModelState(model)

model.debug()
y = model(torch.randn(1,10,64))
(y*y).mean().backward()
model.update()

model.set_drop(0.1,0.0,0.3)
state.layers()
"""