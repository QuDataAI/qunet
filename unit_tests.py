import os, gc, sys, time, datetime, math, random, copy, psutil
import numpy as np,  matplotlib.pyplot as plt, pandas as pd
from   tqdm.auto import tqdm
import torch, torch.nn as nn

from qunet import  ModelState, Trainer, Config, Change, Data, MLP, CNN,  Attention, FFT, Transformer, VitEmb, Vit2d, PointsBlock

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

res = res and CNN.unit_test()
#res = res and Transformer.unit_test()

#m = CNN(input=3, blocks="(c64  m) 2r r128_3_2 r r256_3_2 r r512_3_2 r", shift_after=0)
#m = CNN(input=3, blocks="(cnf64 m)      2r  r128_3_2     r  r256_3_2     r  r512_3_2      r")
#m = CNN(input=3, blocks="r32 r m r64  r r128  r", norm_before=1, norm_inside=1, norm_align=1, norm_block=1,  norm_after=1, res=2)
#m = CNN(input=3, blocks="r64 m r r128 m r r256 m r r512 m", flat=False, avg=False)

#mlp = MLP(input=512, hidden=100, output=10)
#Change.dropout(m, [0.1, 0.2, 0.3])
#Change.shift  (m, [0.1, 0.2, 0.3])

#s = ModelState(m)
#s.layers(2, input_size=(1,3,32,32) ) # 
#s.params()


#s = ModelState(mlp)
#s.layers(2, input_size=(1,512) ) # 

#from torchvision.models import resnet18
#model = resnet18()
#state = ModelState(model)
#state.layers(2, input_size=(1,3,224,224))

#m.debug()
#y = m(torch.randn(1,3,32,32))
#(y*y).mean().backward()
#m.update()
#m.plot()