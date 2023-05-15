# [QuNet](README.md)  


## Visualization of the training process

When `fit` has argument `period_plot > 0`, then every `period_plot` a training plot will be displayed.
By default it contains score and loss:


You can customize the appearance of graphs using the following trainer options:

```python
trainer.view = Config(
    w  = 12,                   # plt-plot width
    h  =  5,                   # plt-plot height
    units = Config(
        unit  = 'epoch',       # 'epoch' | 'sample'
        count = 1e6,           # units for number of samples
        time  = 's'            # time units: ms, s, m, h
    ),

    x_min = 0,                 # minimum value in samples on the x-axis (if < 0 last x_min samples)
    x_max = None,              # maximum value in samples on the x-axis (if None - last)

    loss = Config(                                
        show  = True,          # show loss subplot
        y_min = None,          # fixing the minimum value on the y-axis
        y_max = None,          # fixing the maximum value on the y-axis
        ticks = None,          # how many labels on the y-axis
        lr    = True,          # show learning rate
        labels= True,          # show labels (training events)                
        trn_checks = False,    # show the achievement of the minimum training loss (dots)
        val_checks = True      # show the achievement of the minimum validation loss (dots)
    ),            
    score = Config(                                
        show  = True,          # show score subplot    
        y_min = None,          # fixing the minimum value on the y-axis
        y_max = None,          # fixing the maximum value on the y-axis
        ticks = None,          # how many labels on the y-axis
        lr    = True,          # show learning rate                
        labels = True,         # show labels (training events)
        trn_checks = False,    # show the achievement of the optimum training score (dots)
        val_checks = True      # show the achievement of the optimum validation score (dots)
    ),
)
```

You can change one parameter:
```python
trainer.view.loss.lr = False   # do not show learning rate on loss plot
```
or immediately a group of parameters:
```python
trainer.view.units(unit='sample', count=1e3, time='m')
```

<hr>
