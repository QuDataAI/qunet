# QuNet

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI version](https://badge.fury.io/py/torchinfo.svg)](https://badge.fury.io/py/torchinfo)


Easy working with deep learning models.
* Large set of custom modules for neural networks (MLP, CNN, Transformer, etc.)
* Trainer class for training the model.
* Various tools for visualizing the training process and the state of the model.
* Training large models: float16, mini-batch splitting if it does not fit in memory, etc.

<hr>

## Install

```
pip install qunet
```
<hr>

## Usage

```python
from qunet import Data, Trainer, ExpScheduler

# 1. create dataset
X = torch.rand(10000,1)               
Y = 2*X + 1
data_trn = Data((X,Y), batch_size=128, shuffle=True)

# 2. create trainer, optimizer and scheduler (if need)                                               
tariner = Trainer(model, data_trn)    
trainer.set_optimizer( torch.optim.SGD(model.parameters(), lr=1e-2) )
trainer.set_scheduler( ExpScheduler(lr1=1e-5, lr2=1e-4, samples=100e3) )

# 3. run training
trainer.run(epochs=100, period_plot=5)
```

<hr>

## Model

Model must be a class (successor of nn.Module) with methods:
* The `forward` function takes input `x` and returns output `y`.
These can be tensors or tuples (lists) of tensors.
* The `metrics` function takes  `(x, y_true, y_pred)` and returns the model's scalar loss and tensor quality metric: 
 `(B,1)` for one metric (accuracy, for example) or `(B,n)` for n quality metrics.

For example, for 1D linear regression  $y=f(x)$ with mse-loss and metric as |y_pred-y_true|, model looks like:
```python
class Model(nn.Module):
    def __init__(self):        
        super().__init__() 
        self.fc = nn.Linear( 1, 1 )

    def forward(self, x):                            # (B,1)
        return self.fc(x)                            # (B,1)

    def metrics(self, x, y_pred, y_true)             # (B,1) (B,1)  (B,1)        
        loss   = (y_pred - y_true).pow(2).mean()     # ()     scalar!
        errors = torch.abs(y_pred.detach()-y_true)   # (B,1)  one metric
        return loss, errors                          # ()  (B,1)
```

**Attention**: If the output of the model is one, after the Linear layer the tensor has the shape (B,1).
Therefore, the target data must also have the form (B,1), otherwise we will get an incorrect loss.
```python
X,Y = torch.arange(5).view(-1,1).to(torch.float32),  torch.arange(5).to(torch.float32)
loss = (X-Y).pow(2).mean()  # 4 так как (B,1) - (B,) = (B,1) - (1,B) = (B,B)
```
<hr>

## Data

The `Data` - training or validation data class. It can be overridden or pytorch DataLoader can be used.
Iterator `__next__` of the `Data`  must return an `X,Y` tuple, where:
* X - tensor or tuple (list) of tensors for model input,
* Y - tensor or tuple (list) of tensors for model target values.

For example, let's create training data in which two tensors X1,X2 are the input of the model and one tensor Y is the output (target):
```python    
X1, X2 = np.rand(1000,3), np.rand(1000,3,20)
Y = X1 * torch.Sigmoid(X2).mean(-1)

data_trn = Data( dataset=( (X1,X2), Y ) )  
```        
The data minibatch tuple (X,Y) is used in the Trainer as follows:
```python
for b (X,Y_true) in enumerate(data):  # при обучении
    X, Y_true = to_device(X), to_device(Y_true)            
    Y_pred = model(X)
    loss, score = model.metrics(X, Y_pred, Y_true)
```
So dataset is a list or tuple of two elements (input and target).
Each element can be a tensor or a list (tuple) of tensors.
All tensors in the dataset are assumed to have the same length (by first index).

The Data class constructor has the following parameters:
```python
Data(dataset, shuffle=True, batch_size=64,  whole_batch=False, n_packs=1)
```
* `dataset` - model input and output tuple (X, Y), as described above
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

mnist = datasets.MNIST(root='data', train=True,  transform=ToTensor(), download=True)
data_trn  = DataLoader(dataset=mnist, batch_size=1024, shuffle=True)
```
<hr>

## Trainer

The Trainer is given the model, training and validation data.
Using the `set_optimizer` function, the optimizer is set.
After that, the function `run` is called:
```python
trainer = Trainer(model, data_trn, data_val)
trainer.set_optimizer( torch.optim.SGD(model.parameters(), lr=1e-2) )
trainer.run(epochs=100, pre_val=True, period_plot=10)
```
You can add different training schedulers, customize the output of training graphs, manage the storage of the best models and checkpoints, and much more.

```python
trainer = Trainer(model, data_trn, data_val, device=None, dtype=torch.float32, score_max=False)
```

* `model`     - model for traininig;
* `data_trn`  - training data (Data or DataLoader instance);
* `data_val`  - data for validation (instance of Data or DataLoader); may be missing;
* `score_max` - consider that the metric (the first column of the second tensor returned by the function `metrics` of the model ); should strive to become the maximum (for example, so for accuracy).

Other properties of `Trainer` allow you to customize the appearance of graphs, save models, manage training, and so on.
They will be discussed in the relevant sections.

```python
trainer.run(epochs=None, samples=None,
            pre_val=False,  period_val=1, period_plot=100, period_checks=1,          
            period_val_beg = 4, samples_beg = None)
```

* `epochs`         - number of epochs for training (passes of one data_trn pack). If not defined (None) works "infinitely".
* `samples`        - if defined, then training will stop after this number of samples, even if epochs has not ended
* `pre_val`        - validate before starting training
* `period_val`     - period after which validation run (in epochs)
* `period_plot`    - period after which the training plot is displayed (in epochs)
* `period_checks`  - period after which checkpoints are made and the current model is saved (in epochs)
* `period_val_beg` - validation period on the first `samples_beg` samples. Used when validation needs to be done less frequently at the start of training.
* `samples_beg`   -  the number of samples from the start, after which the validation period will be equal to `period_val`

<hr>

## Visualization of the training process

<img src="img/loss.png">


```python
trainer.view = {                    
    'w'            : 12,         # plt-plot width
    'h'            :  5,         # plt-plot height

    'count_units'  : 1e6,        # units for number of samples
    'time_units'   : 's',        # time units: ms, s, m, h

    'x_min'        : 0,          # minimum value in samples on the x-axis (if < 0 last x_min samples)
    'x_max'        : None,       # maximum value in samples on the x-axis (if None - last)

    'loss': {                                
        'show'  : True,          # show loss subplot
        'y_min' : None,          # fixing the minimum value on the y-axis
        'y_max' : None,          # fixing the maximum value on the y-axis
        'ticks' : None,          # how many labels on the y-axis
        'lr'    : True,          # show learning rate
        'checks': True,          # show the achievement of the minimum loss (dots)
        'labels': True,          # show labels (training events)
    },

    'score': {                    
        'show'  : True,          # show score subplot    
        'y_min' : None,          # fixing the minimum value on the y-axis
        'y_max' : None,          # fixing the maximum value on the y-axis
        'ticks' : None,          # how many labels on the y-axis
        'lr'    : True,          # show learning rate
        'checks': True,          # show the achievement of the optimum score (dots)
        'labels': True,          # show labels (training events)
    }
}
```


<hr>

## Using Schedules

Schedulers allow you to control the learning process by changing the learning rate according to the required algorithm.
There can be one or more schedulers. In the latter case, they are processed sequentially one after another.
Существуют следующие шедулеры:
* `LineScheduler(lr1, lr2, samples)` - changes the learning rate from `lr1` to `lr2` over `samples` training samples. If `lr1` is not specified, the optimizer's current lr is used for it.
* `ExpScheduler(lr1, lr2, samples)` - similar, but changing `lr` from `lr1` to `lr2` is exponential.
* `CosScheduler(lr1, lr_hot,  lr2, samples, warmup)` - changing `lr` by cosine with preliminary linear heating during `warmup` samples from `lr1` to `lr_hot`.
* `WaitScheduler(lr1, samples)` - wait for `samples` samples with unchanged `lr` (as usual, the last value is taken if `lr1` is not set). This scheduler is useful when using lists of schedulers.

Each scheduler has a `plot` method that can be used to display the training plot:
```python
sch = CosScheduler(lr1=1e-5, lr_hot=1e-2, lr2=1e-4,  samples=100e3, warmup=1e3)
sch.plot(log=True)
```
You can also call the `trainer.plot_schedulers()` method of the `Trainer` class.
It will draw the schedule of the list of schedulers added to the trainer.

Compiling a list of schedulers is done by the following methods of the `Trainer` class:
* `set_scheduler`( sch ) - set a list of schedulers from one scheduler sch (after clearing the list);
* `add_scheduler`( sch ) - add scheduler sch
* `del_scheduler`(i)     - remove the i-th scheduler from the list (numbering from zero)

This group of methods works with all schedulers:
* `reset_schedulers`() - reset all scheduler counters and make them active (starting from the first one)
* `stop_schedulers` () - stop all schedulers
* `clear_schedulers`() - clear list of schedulers
Example:


<img src="img/schedulers.png">
<hr>

## Checkpoints and best model

<hr>

## Batch argumentation

<hr>


## Working with large models

<hr>

## Model state visualization

<hr>

## Examples

* Regression_1D - visualization of changes in model parameters
* Interpolation_F(x)
* MNIST
* Vanishing gradient

<hr>

## Versions

* 0.0.4 - fixed version for competition IceCube (kaggle)


$$
E=mc^2
$$
