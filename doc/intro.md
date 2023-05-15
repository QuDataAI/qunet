# [QuNet](README.md) - Quick start

## Install

```
pip install qunet
```
<hr>


## Import

```python
import torch, torch.nn as nn
import matplotlib.pyplot as plt

from qunet import  Data, MLP, Trainer, Callback
```

<img src="img/bce-data.png" style="float:right; width:350px;">

## Toy data

Consider a two-dimensional feature space. There are objects of two kinds (classification into two classes).<br>
Let's generate 1200 objects and draw them in the feature space.

```python
n_samples = 1200
X = 2*torch.rand(n_samples, 2) - 1      # [-1...1]^2
Y = torch.sum(X**2, axis=1) < 0.5       # inside or outside circle
Y = Y.float().view(-1,1)                # float and shape (N,1) for MSE or BCELoss, not (N,) !!!

fig = plt.figure(figsize=(4,4))
plt.scatter(X[:,0], X[:,1], c=Y, s=3, cmap="bwr")
plt.grid(ls=":")
plt.show()
```

Then we use `Data` - data loader from the `QuNet` library.<br>
All data will be divided into training (`data_trn`) and validation ('data_val'). Training data will be shuffled before each epoch (`shuffle=True`)
```python
n_trn = int(n_samples*0.8)
data_trn = Data( (X[:n_trn], Y[:n_trn]), batch_size=128, shuffle=True)
data_val = Data( (X[n_trn:], Y[n_trn:]), batch_size=256 )
```
Note that the `Data` class constructor is passed a tuple (you can list) of the data needed for training. In our case, it is `(X, Y)`. In the same sequence, these data will be present in each batch. The batch size `batch_size` can be changed later by simply assigning `data_trn.btach_size = 64`


## Create Model

To work with the trainer, the model must be the successor of nn.Module.
In the constructor, we will create a fully connected neural network with two inputs, one output, and one hidden layer with 16 neurons.
To do this, we will use the `MLP` module (Multilayer Perceptron) of the `QuData` library.

```python
class Model(nn.Module):
    def __init__(self):              
        super().__init__() 
        self.mlp  = MLP(input=2, hidden=[16], output=1,  drop=0.1)
        self.loss = nn.BCELoss()

    def forward(self, x):                                 # (B,2)
        x = self.mlp(x)
        return torch.sigmoid(x)                           # (B,1)

    def training_step(self, batch, batch_id):        
        x, y_true = batch                                 # the model knows the minbatch format
        y_pred = self(x)                                  # (B,1)  forward function call

        loss  = self.loss(y_pred, y_true)                 # loss for optimization (scalar)!  

        y_pred = (y_pred.detach() > 0.5)                  # number of predicted class
        acc = (y_pred == y_true.bool()).float().mean()    # accuracy        

        return {'loss': loss, 'score': acc}                # if no score, you can return loss

model = Model()
```

In addition to forward, you should additionally define some methods that the trainer uses. В минимальном случае это должен быть метод `training_step`. It accepts a batch of examples as input, consisting of a pair of `x` (model input) and `y` (target value = 0 or 1).
The output should be a dictionary containing an 'loss' (it will be minimized). If desired, you can return 'score' with various quality indicators of the model (below, there is only one such indicator - accuracy).

<hr>

## Trainer

The Trainer is given the model, training and validation data.
Using the `set_optimizer` function, the optimizer is set.
After that, the function `fit` is called:
```python
trainer = Trainer(model, data_trn, data_val)

trainer.set_optimizer( torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) )
trainer.fit(epochs=500, period_plot=100)
```
You can add different training schedulers, customize the output of training graphs, manage the storage of the best models and checkpoints, and much more. 
In particular, the parameter `period_plot = 100` means that every 100 epochs the training plot will be displayed:

<img src="img/loss.png">

<hr>

<img src="img/bce-res.png" style="float:right; width:350px;">

## Callback

```python
class Visual(Callback):
    def on_after_plot(self, trainer, model):
        """ Called after training plot. """        
        h, w = 64, 64
        x = torch.cartesian_prod(torch.linspace(-1,1,h), torch.linspace(-1,1,w)) 
        model.train(False)
        with torch.no_grad():
            y = model(x).view(h,w)

        fig = plt.figure(figsize=(4,4))
        plt.imshow(y, cmap="bwr", origin='lower', extent=[-1,1,-1,1], alpha=0.5)
        plt.scatter(X[:,0], X[:,1], c=Y, s=3, cmap="bwr")
        plt.show()            
```

...

```python
trainer.callbacks=[ Visual() ]
trainer.set_optimizer( torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) )
trainer.fit(500, period_plot=100)
```

<hr>


## Using Schedules

Schedulers allow you to control the learning process by changing the learning rate according to the required algorithm.
There can be one or more schedulers. In the latter case, they are processed sequentially one after another.
There are the following schedulers:
* `Scheduler_Line(lr1, lr2, epochs)` - changes the learning rate from `lr1` to `lr2` over `epochs` training epochs. If `lr1` is not specified, the optimizer's current lr is used for it.
* `Scheduler_Exp(lr1, lr2, epochs)` - similar, but changing `lr` from `lr1` to `lr2` is exponential.
* `Scheduler_Cos(lr1, lr_hot,  lr2, epochs, warmup)` - changing `lr` by cosine with preliminary linear heating during `warmup` epochs from `lr1` to `lr_hot`.
* `Scheduler_Const(lr1, epochs)` - wait for `epochs` epochs with unchanged `lr` (as usual, the last value is taken if `lr1` is not set). This scheduler is useful when using lists of schedulers.

Instead of `epochs`, you can specify `samples` (number of training samples)

Each scheduler has a `plot` method that can be used to display the training plot:
```python
sch = Scheduler_Cos(lr1=1e-5, lr_hot=1e-2, lr2=1e-4,  samples=10e3,  warmup=1e3)
sch.plot(log=True, samples=20e3, epochs=100)
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

Example of learning curves of various schedulers:

<img src="img/schedulers.png">

An example of using schedulers can be found in:
<a href="https://colab.research.google.com/drive/179sHb3WyHNrSJKGLfKrXaAzvShmS1SSf?usp=sharing">notebook Interpolation_F(x)</a> 

<hr>

## Best Model and Checkpoints

Trainer can store the best models in memory or on disk.
Small models are convenient to keep in memory. 
When the best validation loss or score is reached, a copy of the model is made. 
To do this, you need to enable `train.best.copy` and specify the target value for which you want to remember the model in the `monitor` list:
```python
trainer.best(copy=True)
trainer.fit(epochs=200, monitor=['score'])
trainer.save("best_score.pt", trainer.best.score_model)
```
The last best model will be in `trainer.best.loss_model` and `trainer.best.score_model`.
The values of the corresponding metrics are in `trainer.best.loss` and `trainer.best.score`.
These models can be used to roll back if something went wrong:
```python
trainer.model = copy.deepcopy(trainer.best.score_model)   
```
To save the best models by loss and/or score on disk, you need to set folders.
Saving will occur if you specify `monitor` in `fit`:
```python
trainer.folders(loss='log/loss', score='log/loss',  points='log/checkpoints')
trainer.fit(epochs=200, monitor=['score', 'loss', 'points'], period_points=10)
```
The best model by score and loss will be saved each time a new best value is reached.
Checkpoints (`points`) are simply saving the current state of the model.
They can be done with the desired periodicity in epochs (period_points=1 by default).

The best score is the metric of the first element in the score.
If `trainer.score_max=True`, then the higher the score, the better (for example, accuracy).
<hr>

