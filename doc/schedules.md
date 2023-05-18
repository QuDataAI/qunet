# [QuNet](README.md) - Using Schedules


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


