# [QuNet](README.md)  Model


## Model

Model must be a class (successor of nn.Module) with functions:

* `forward(x)` function takes input `x` and returns output `y`. 
This function is not used directly by the coach and usually contains the complete logic of the model.
* `training_step(batch, batch_id)` - called by the trainer during the training phase. 
Should return a scalar loss (with computational graph).
It can also return a dictionary like `{"loss": loss, "score": torch.hstack([accuracy, tnr, tpr])}`, where score is a quality metrics.
* `validation_step(batch, batch_id)` - similarly called at the model validation stage.
If it does the same calculations as `training_step`, it can be omitted.
* `predict_step(batch, batch_id)` - required when using the `predict` method. Should return a tensor `y_pred` of the model's output (or a dictionary `{"output": y_pred, "score": metrics}`, where `metrics` are any quality metrics tensor for each example).

For example, for 1D linear regression  $y=f(x)$ with mse-loss and metric as |y_pred-y_true|, model looks like:
```python
class Model(nn.Module):
    def __init__(self):              
        super().__init__() 
        self.fc = nn.Linear( 1, 1 )

    def forward(self, x):                                 # (B,1)
        return self.fc(x)                                 # (B,1)

    def training_step(self, batch, batch_id):        
        x, y_true = batch                                 # the model knows the minbatch format
        y_pred = self(x)                                  # (B,1)  forward function call
        loss  = (y_pred - y_true).pow(2).mean()           # ()     loss for optimization (scalar)!
        error = torch.abs(y_pred.detach()-y_true).mean()  # (B,1)  error for batch samples
        return {'loss':loss, 'score': error}              # if no score, you can return loss
```

