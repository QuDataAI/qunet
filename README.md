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
from qunet import Data, Trainer, Scheduler_Exp

# 1. create dataset
X = torch.rand(10000,1)               
Y = 2*X + 1
data_trn = Data((X,Y), batch_size=128, shuffle=True)

# 2. create trainer, optimizer and scheduler (if need)                                               
tariner = Trainer(model, data_trn)    
trainer.set_optimizer( torch.optim.SGD(model.parameters(), lr=1e-2) )
trainer.set_scheduler( Scheduler_Exp(lr1=1e-5, lr2=1e-4, samples=100e3) )

# 3. run training
trainer.fit(epochs=100, period_plot=5)
```

<hr>

## Model

Model must be a class (successor of nn.Module) with functions (only `training_step` is required):

* `forward(x)` function takes input `x` and returns output `y`. 
This function is not used directly by the coach and usually contains a description of the model.
* `training_step(batch, batch_id)` - called by the trainer during the training phase. 
Should return a scalar loss (with computational graph).
It can also return a dictionary like `{"loss": scalar, "score": tensor}`, where score is a quality metric.
* `validation_step(batch, batch_id)` - similarly called at the model validation stage.
If it does the same calculations as `training_step`, it can be omitted.
* `predict_step(batch, batch_id)` - required when using the predict method. Should return a tensor of the model's work.

For example, for 1D linear regression  $y=f(x)$ with mse-loss and metric as |y_pred-y_true|, model looks like:
```python
class Model(nn.Module):
    def __init__(self):      
        """ Creating Model Parameters """  
        super().__init__() 
        self.fc = nn.Linear( 1, 1 )

    def forward(self, x):                            # (B,1)
        """ Defining Model Calculations """
        return self.fc(x)                            # (B,1)

    def training_step(self, batch, batch_id):
        """ Called by the trainer during the training step """
        x, y_pred = batch                            # the model knows the minbatch format
        y_true = self(x)                             # (B,1)  forward function call
        loss   = (y_pred - y_true).pow(2).mean()     # ()     loss for optimization (scalar)!        
        errors = torch.abs(y_pred.detach()-y_true)   # (B,1)  errors for each sample (one metric)
        return {'loss':loss, 'score': errors}        # if there is no score, you can simply return loss
```
As we can see, the model description interface is the same as the library interface <a href="https://lightning.ai/">Pytorch Lightning</a>

<hr>

## Data

The `Data` - training or validation data class. It can be overridden or pytorch DataLoader can be used.
Iterator `__next__`  must return an mini-batch, the same structure as passed `dataset` when creating the Data.
For example, let's create training data in which two tensors X1,X2 are the input of the model and one tensor Y is the output (target):
```python    
X1, X2 = np.rand(1000,3), np.rand(1000,3,20)
Y = X1 * torch.Sigmoid(X2).mean(-1)

data_trn = Data( dataset=( (X1,X2), Y ) )  
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
<hr>

## Trainer

The Trainer is given the model, training and validation data.
Using the `set_optimizer` function, the optimizer is set.
After that, the function `fit` is called:
```python
trainer = Trainer(model, data_trn, data_val)
trainer.set_optimizer( torch.optim.SGD(model.parameters(), lr=1e-2) )
trainer.fit(epochs=100, pre_val=True, period_plot=10)
```
You can add different training schedulers, customize the output of training graphs, manage the storage of the best models and checkpoints, and much more.

```python
trainer = Trainer(model, data_trn, data_val, device=None, dtype=torch.float32, score_max=False)
```

* `model`     - model for traininig;
* `data_trn`  - training data (Data or DataLoader instance);
* `data_val`  - data for validation (instance of Data or DataLoader); may be missing;
* `score_max` - consider that the metric (the first column of the tensor `score` returned by the function `training_step` of the model); should strive to become the maximum (for example, so for accuracy).

Other properties of `Trainer` allow you to customize the appearance of graphs, save models, manage training, and so on.
They will be discussed in the relevant sections.

```python
trainer.fit(epochs=None,   samples=None,            
            pre_val=False, period_val=1, period_plot=100,         
            period_checks=1, period_val_beg = 4, samples_beg = None,
            period_call:int = 0, callback = None):     
```

* `epochs`         - number of epochs for training (passes of one data_trn pack). If not defined (None) works "infinitely".
* `samples`        - if defined, then training will stop after this number of samples, even if epochs has not ended
* `pre_val`        - validate model before starting training
* `period_val`     - the period with which the validation model runs (in epochs)
* `period_plot`    - the period with which the training plot is displayed  (in epochs)
* `period_call`    - callback custom function call period
* `callback`       - custom function called with period_info
* `period_checks`  - the period with which the checkpoints are made and the current model is saved (in epochs)
* `period_val_beg` - the period with which the validation model runs on the first `samples_beg` samples. Used when validation needs to be done less frequently at the start of training.
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
        'labels': True,          # show labels (training events)
        'trn_checks': True,      # show the achievement of the minimum training loss (dots)
        'val_checks': True,      # show the achievement of the minimum validation loss (dots)
    },

    'score': {                    
        'show'  : True,          # show score subplot    
        'y_min' : None,          # fixing the minimum value on the y-axis
        'y_max' : None,          # fixing the maximum value on the y-axis
        'ticks' : None,          # how many labels on the y-axis
        'lr'    : True,          # show learning rate
        'labels': True,          # show labels (training events)
        'trn_checks': True,      # show the achievement of the optimum training score (dots)                
        'val_checks': True,      # show the achievement of the optimum validation score (dots)                
    }
}
```


<hr>

## Using Schedules

Schedulers allow you to control the learning process by changing the learning rate according to the required algorithm.
There can be one or more schedulers. In the latter case, they are processed sequentially one after another.
There are the following schedulers:
* `Scheduler_Line(lr1, lr2, samples)` - changes the learning rate from `lr1` to `lr2` over `samples` training samples. If `lr1` is not specified, the optimizer's current lr is used for it.
* `Scheduler_Exp(lr1, lr2, samples)` - similar, but changing `lr` from `lr1` to `lr2` is exponential.
* `Scheduler_Cos(lr1, lr_hot,  lr2, samples, warmup)` - changing `lr` by cosine with preliminary linear heating during `warmup` samples from `lr1` to `lr_hot`.
* `Scheduler_Const(lr1, samples)` - wait for `samples` samples with unchanged `lr` (as usual, the last value is taken if `lr1` is not set). This scheduler is useful when using lists of schedulers.

Each scheduler has a `plot` method that can be used to display the training plot:
```python
sch = Scheduler_Cos(lr1=1e-5, lr_hot=1e-2, lr2=1e-4,  samples=100e3, warmup=1e3)
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

## Best Model and Checkpoints

If you set these flags to `True`, then `Trainer` will save in memory the last best model by validation score or/and loss:
```python
trainer.copy_best_score_model = True  # to copy the best model by val score
trainer.copy_best_loss_model  = True  # to copy the best model by val loss
```
These models can be used to roll back if something went wrong:
```python
train.model = copy.deepcopy(trainer.best_score_model)   
```

If the following folders are defined (by default `None`), then the best model by validation loss, score will be saved on disk and intermediate versions of the model will be saved with the period `period_checks` (argument of `fit` function).
```python
trainer.folder_loss   = "log/best_loss"   # folder to save the best val loss models
trainer.folder_score  = "log/best_score"  # folder to save the best val score models
trainer.folder_checks = "log/checkpoints" # folder to save checkpoints        
```
The best score is the metric of the first column of the second return tensor in the metrics function of the model.
If `trainer.score_max=True`, then the higher the score, the better (for example, accuracy).
<hr>

## Batch Argumentation

<hr>


## Working with the Large Models

<hr>

## Model State Visualization

<hr>

## Examples

* <a href="https://colab.research.google.com/drive/179sHb3WyHNrSJKGLfKrXaAzvShmS1SSf?usp=sharing">Interpolation_F(x)</a> - interpolation of a function of one variable (example of setting up a training plot; working with the list of schedulers; adding a custom plot)
* <a href="https://colab.research.google.com/drive/1N4b6mwUvH-o-t6VIiuhq7FMuGRabdOm0?usp=sharing">MNIST</a> - recognition of handwritten digits 0-9 (example using pytorch DataLoader, model predict, show errors, confusion matrix)
* <a href="https://colab.research.google.com/drive/1ThxnMrAjuFTGKXLI-93oRa9doNpP32y4?usp=sharing">CIFAR10</a>  (truncated EfficientNet, pre-trained parameters, bone freezing, augmentation)
* Vanishing gradient
* Regression_1D - visualization of changes in model parameters

<hr>

## Versions

* 0.0.4 - fixed version for competition IceCube (kaggle)


$$
E=mc^2
$$
