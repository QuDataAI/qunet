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

