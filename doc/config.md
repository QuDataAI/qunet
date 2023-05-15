# [QuNet](README.md) - Config

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
