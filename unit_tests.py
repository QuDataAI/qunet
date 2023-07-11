import os, gc, sys, time, datetime, math, random, copy, psutil
import numpy as np,  matplotlib.pyplot as plt, pandas as pd
from   tqdm.auto import tqdm
import torch, torch.nn as nn

from qunet import  ModelState, Trainer, Config, Change, Data, MLP, ResMLP, CNN,  Attention, FFT, TransformerBlock, Transformer, VitEmb, Vit2d, PointsBlock

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

print("--------------------------")
print("unit tests result = ", res)
print("--------------------------")
"""
"""
                                                       ops            pars
w = 32          16           8           4       2
blocks = "r128   m    r256   m    r512   m"            624,683    4,742,403        
blocks = "r128   m r  r256   m r  r512   m"            775,675    6,217,733
blocks = "r128   m r  r256_3_2 r  r512_3_2"            423,355    6,217,733 <---
blocks = "r128_3_2 r  r256_3_2 r  r512_3_2"            307,160    6,217,733

blocks = "cnf64 r m  r128_3_2 2r r256_3_2 2r  r512_3_2"  196,798    7,846,856
              2 74
""" 
#m = CNN(input=3, blocks="(c64  m) 2r r128_3_2 r r256_3_2 r r512_3_2 r", shift_after=0)
m = CNN(input=3, blocks="(cnf64 m)      2r  r128_3_2     r  r256_3_2     r  r512_3_2      r")
#m = CNN(input=3, blocks="r64 m r r128 m r r256 m r r512 m", flat=False, avg=False)

#mlp = MLP(input=512, hidden=100, output=10)
#Change.dropout(m, [0.1, 0.2, 0.3])
#Change.shift  (m, [0.1, 0.2, 0.3])

#s = ModelState(m)
#s.layers(2, input_size=(1,3,32,32) ) # 


#s = ModelState(mlp)
#s.layers(2, input_size=(1,512) ) # 

#from torchvision.models import resnet18
#model = resnet18()
#state = ModelState(model)
#state.layers(2, input_size=(1,3,224,224))

m.debug()
y = m(torch.randn(1,3,32,32))
(y*y).mean().backward()
m.update()

m.plot()