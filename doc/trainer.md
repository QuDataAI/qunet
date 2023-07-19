# [QuNet](README.md) - Trainer


## Trainer constructor

When creating a trainer, the following parameters can be passed to it:
* `model`    (nn.Module): model for traininig; must have a training_step method
* `data_trn` (Data or DataLoader): training data
* `data_val`  (Data or DataLoader): data for validatio; may be missing;
* `callbacks` (Callback): list of Callback instances to be called on events
* `score_max` (bool): consider that the metric (the first column of the second tensor returned by the function `metrics` of the model ); should strive to become the maximum (for example, so for accuracy).
* `wandb_cfg` = Config(
    - `api_key` (str) WANDB API key
    - `project_name` (str) WANDB project name
    - `run_name`(str, optional) WANDB run name
    - `ema` (bool) enable EMA (Exponential Moving Average) support
    - `ema_decay` (float) average decay for EMA
    - `ema_start_epoch` (int) first epoch to average weights (avoid first random values impact)


<hr>

## Train model
Function `fit` uses for run model training with next parameters:
* `epochs` (int) number of epochs for training (passes of one data_trn pack). If not defined (None) works "infinitely".
* `samples` (int) if defined, then will stop after this number of samples, even if epochs has not ended
* `accumulate_grad_batches` (int=1) accumulate gradients from that number of batches before update model parameters 
* `period_plot` (int=0) period after which the training plot is displayed (in epochs)
* `pre_val` (bool=False) validate before starting training
* `period_val_beg` (int=1) validation period on the first val_beg epochs
* `val_beg` (int=None) the number of epochs from the start, after which the validation period will be equal to period_val.
* `period_point` (int=1) period after which checkpoints  are made(in epochs)
* `point_start` (int=1) period after fit runs will be saved checkpoints with period period_points (in epochs)
* `monitor` (list=[]) what to save in folders: monitor=['loss'] or monitor=['loss', 'score', 'point'] 
* `patience` (int) after how many epochs to stop if there was no better loss, but a better score during this time
* `state` (ModelState=None) an instance of the ModelState class that accumulates information about gradients on model parameters
* `period_state` (int=0) period after which the model state  plot is displayed (in epochs)               

Example
```
  tariner = Trainer(model, data_trn=data_trn, data_val=data_val, score_max=True)
  trainer.view.units(count=1e6, time='m');
  trainer.best(copy=True)
  trainer.set_optimizer( torch.optim.Adam(model.parameters(), lr=1e-3) )
  trainer.fit(epochs=200, period_plot = 50, monitor=['score'], patience=100)
```
