import os, gc, sys, time, datetime, math, random, copy, psutil
import numpy as np,  matplotlib.pyplot as plt, pandas as pd
from   tqdm.auto import tqdm
import torch, torch.nn as nn

from qunet import  ModelState, Trainer, Config, Data, MLP, ResMLP, CNN,  Attention, FFT, TransformerBlock, Transformer, VitEmb, Vit2d, PointsBlock

print(torch.__version__)
print(torch.cuda.is_available())
print("--------------------------")

res = True
"""
res = res and MLP.unit_test()
res = res and ResMLP.unit_test()
res = res and CNN.unit_test()
res = res and Attention.unit_test()
res = res and FFT.unit_test()
res = res and TransformerBlock.unit_test()
res = res and Transformer.unit_test()
res = res and VitEmb.unit_test()
res = res and Vit2d.unit_test()
res = res and Data.unit_test()
"""
m = nn.Sequential(
    #CNN(input=1, blocks="cnf128 m d cnf256 m d cnf512 m d"),  # тоже что "c32 n f m c64 n f m"    
    #CNN(input=1, blocks="cnf128_3_2 d cnf256_3_2 d cnf512_3_2  d"),
    CNN(input=1, blocks="(c64_7_2 n f  m3_2) 2r r128_3_2 r r256_3_2 r r512_3_2 r"), 
    MLP(input=512, output=10) )

m[0].set_dropout([0.1, 0.2, 0.3])

s = ModelState(m)
s.layers(2, input_size=(1,1,224,224) )

print("--------------------------")
print("unit tests result = ", res)
print("--------------------------")


#from torchvision.models import resnet18
#model = resnet18()
#state = ModelState(model)
#state.layers(2, input_size=(1,3,224,224))


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