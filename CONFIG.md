# QuNet - Config

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI version](https://badge.fury.io/py/torchinfo.svg)](https://badge.fury.io/py/torchinfo)

[QuNet](README.md)  uses the `Config` class to store model or `Trainer` parameters.

<hr>

Example:
```python        
cfg = Config(
    x=2, 
    y=5
)
cfg.x = 5
print(cfg)         # x:5, y:5
print(cfg.x)       # 5
```
cfg variables can be instances of `Config`
```python
cfg = Config(
    x=2, 
    y=5,
    z=Config( text='Ok' )
)
```
You can change values for a group of parameters:
```python
cfg(x=5, y=7)
```
In `Config`, you can pass another `Config` as the first argument:
```python
cfg2 = Config(x=0)
cfg(cfg2, y=7)
```
