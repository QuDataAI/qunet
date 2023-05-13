# QuNet - Tasks

| [Linear Regression](#linear-regression) | [Tensor contraction](#tensor-contraction) |
| ---- | ---  |



## Linear Regression
Modeling linear regression $y=x_1+x_2$ with MSE-loss:
```python
# dataset:
X = torch.rand(100,2)
Y = X.sum(dim=1) 

# model:
model = nn.Linear(2,1, bias=False)

Y_pred = model(X)
loss = (Y_pred-Y).pow(2).mean()
```
Where is the mistake made and why does it lead to an incorrect result?

Little hint:
```python
model = nn.Linear(2,1, bias=False)
model.weight.data=torch.ones(1,2)

Y_pred = model(X)
loss = (Y_pred-Y).pow(2).mean()

print(loss.detach()) # ?????
```

<details>
<b>Answer</b> :
We need to change the shape of the target in the training data:

```python
Y = X.sum(dim=1).view(-1,1)      # (B,1)
# or
Y = X.sum(dim=1, keepdims=True)  # (B,1)
```

<b>Explanation</b> :
Input matrix shape `X.shape=(100, 2)`.
The line layer (2,1) produces `Y_pred.shape=(100,1)` as output.
We compare it with `Y.shape=(100,)`.
In pytorch, if the shape of the tensors does not match, a broadcasting procedure is done in which the dimension is added in front. When subtracting, we get:

```python
(100,)-(100,1) -> (1,100)-(100,1)  -> (100,100)-(100,100) 
```

<b>For example</b> :
```python
Y1 = torch.tensor([1, 2])         # (2,)
Y2 = torch.tensor([[1], [2]])     # (2,1)
Y1 - Y2 = 
tensor([[ 0,  1],
        [-1,  0]])
```
</details>

<hr>

## Tensor contraction

Глубокое обучение - это преобразования тензоров. Свёртка (основное преобразование) равна сумме по некоторым индексам произведений компонент тензоров. Например:

```python
X = torch.rand(2,3,4)
Y = torch.rand(4,5)
Z = X @ Y             # (2,3,5):  Z[a,b,c] - это сумма по i в X[a,b,i] * Y[i,c]
```

Необходимо предложить максимальное число способов (без циклов!) свернуть по последнему в X и первому в Y индексу следующие 2 тензора:

```python
X = torch.rand(2,3)
Y = torch.rand(3,4,5)
Z = f(X,Y)              # (2,4,5)
```
</details>
```python
# 1.
torch.einsum('ij,jkl->ikl', X, Y)
# 2.
torch.tensordot(X, Y, dims=([1],[0]) )
# 3.
( X @ Y.view(3,-1) ).view(2,4,5) 
# 4.
( X @ Y.permute(1,0,2) ).permute(1,0,2) 
```