# [QuNet](README.md) - ModelState

```
state = ModelState(model)

state.num_params() # number of model parameters

state.layers()     # model layers
state.params()     # model parameters 
state.state()      # params and register_buffer

state.plot()       # draw params and grads
ModelState.hist_params([m.weight, m.bias], ["w","b"])
```
