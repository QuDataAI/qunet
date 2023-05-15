# [QuNet](README.md)  - Data



QuNet has a Data class - data for model training or validation. It can be overridden or pytorch DataLoader can be used.
The iterator `__next__`  is supposed  must return an mini-batch, has the same structure as passed `dataset` when creating the `Data`.
For example, let's create training data in which two tensors X1,X2 are the input of the model and one tensor Y is the output (target):
```python    
from qunet import Data

X1, X2 = torch.rand(1000,3), torch.rand(1000,3,20)
Y = X1 * torch.sigmoid(X2).mean(-1)

data_trn = Data( dataset=( X1, X2, Y ), batch_size=100)  
 
for x1,x2, y in data_trn:
    print(x1.shape, x2.shape, y.shape)  # (100,3) (100,3,20) (100,3)
```        
All tensors in the dataset are assumed to have the same length (by first index).
The model is responsible for interpreting the composition of the mini-batch.

The Data class constructor has the following parameters:
```python
Data(dataset, shuffle=True, batch_size=64,  whole_batch=False, n_packs=1)
```
* `dataset` - model training data: tensor X or tuple input and output tensors: (X, Y), and etc.
* `shuffle` - shuffle data after after passing through all examples
* `batch_size` - minibatch size; can be changed later: data_trn.batch_size = 1024
* `whole_batch` - return minibatches of batch_size only; if the total number of examples is not divisible by batch_size, you may end up with one small batch with an unreliable gradient. If whole_batch = True, such a batch will not be issued.
* `n_packs` - data is split into n_packs packs; the passage of one pack is considered an training ephoch. It is used to a large dataset, when it is necessary to do validation more often.
</ul>

You can also use the standard DataLoader with Trainer:
```python
from torchvision            import datasets
from torchvision.transforms import ToTensor 
from torch.utils.data       import DataLoader

mnist    = datasets.MNIST(root='data', train=True,  transform=ToTensor(), download=True)
data_trn = DataLoader(dataset=mnist, batch_size=1024, shuffle=True)
```