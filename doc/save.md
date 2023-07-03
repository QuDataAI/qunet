# [QuNet](README.md) - Save and Load models

Текущая модель trainer.model сохраняется при помощи метода тренера `save`:
```python
trainer.save("name.pt")           # save current model 
```
Вместе с моделью также сохраняется состояние оптимизатора, конфигурационный файл модели и история её обучения.
Поэтому при загрузке модели возвращается экземпляр класса `Trainer`, содержащий все эти данные.
Статическому методу `load` необходимо передать  класс модели
(он должнен быть объявлен в текущем модуле):
```python
class Model(nn.Module):
    ...

trainer = Trainer.load('name.pt', Model)
```

Только история обучения (без модели) загружается при помощи метода `load_hist`.
После этого историю можно построить в виде графиков:
```python
hist = Trainer.load_hist("cur_model.pt")
trainer.plot(hist)
```

<hr>

Обычно сохранение модели происходит автоматически по достижению лучших валидационных `loss` или/и `score`, а также как чекпоинты (`point`) модели с некоторой периодичностью. 
Чтобы это начало происходить, при вызове метода `fit`, в параметре `monitor` следует указать список того, что необходимо сохранять:
```python
trainer.fit(epochs=100,  monitor=['score', 'point'])
```
По умолчанию, чекпоинты сохраняются с первой эпохи и каждую эпоху.
Это можно изменить при помощи параметров `point_start` (с какой эпохи начинать сохранять) и `period_point` (с какой периодичностью в эпохах сохранять):
```python
trainer.fit(epochs=100,  monitor=['point'], point_start=100, period_point=10)
```

Пути по которым сохраянются модели определяются в свойстве тренера `folders`.
По умолчанию это:
```python
trainer.folders(
        loss   =    "models/loss",      # folder to save the best val loss models
        score  =    "models/score",     # folder to save the best val score models
        loss_ema =  "models/loss_ema",  # folder to save the best val loss EMA models
        score_ema = "models/score_ema", # folder to save the best val score EMA models
        point  =    "models/point",     # folder to save checkpoints
        prefix =    "")                 # add prefix to file name 
```
При желании их можно переопределить:
```python
trainer.folders(loss="exper01/loss", score="exper01/score")
```
<hr>

При работе со сложной моделью, обычно, необходимы эксперименты по модификации её архитектуры.
Все параметры, влияющие на архитектуру, имеет смысл вынести в конфиг модели.
Этот конфиг будет сохраняться вместе в моделью, если его имя переменной конфига - `cfg`  (!!!).
При загрузке он передаётся в модель как аргумент.
Поэтому рекомендуется следующая организация конструктора модели:
```python
class Model(nn.Module):
    def __init__(self, *arg):        
        super().__init__()       

        self.cfg = Config(
            # model parameters
        )
        cfg = self.cfg(*arg)       # modify parameters via constructor argument

        # create a model
        self.mlp = MLP(hidden = cfg.hidden)
        # ...
```
Cсылка `cfg` на ` self.cfg` делается только для сокращения дальнейшего доступа к параметрам (`cfg.hidden` вместо `self.cfg.hidden`). Передача аргумента конструктора: `self.cfg(*arg)`, менят заданные по умолчанию параметры (это и произойдёт в загрузчике).

Можно также добавить в конструктор пары ключ-значение:
```python
class Model(nn.Module):
    def __init__(self, *arg, **kvargs):        
        super().__init__()       

        self.cfg = Config(
            par1 = 1,
            par2 = 2,
            par3 = 3
        )
        cfg = self.cfg(*arg, **kvargs)    # modify parameters via constructor arguments

        # ... create a model

#model = Model(par2=4,  par3=8)   # эксперимент 1
model  = Model(par2=20, par3=30)  # эксперимент 2
```
В этом случае `model.cfg` будет содержать параметры `par1=1, par2=20, par3=30`.
Конфиг `model.cfg` сохранится вместе с моделью, если его имя cfg.
При загрузке будет создан экземпляр `model=Model(cfg)` и восстановлены все эти параметры.

<hr>

Ниже приведен полный пример процесса сохранения и загрузки модели:
```python
from qunet import Config, Data, Trainer, MLP, ModelState
import torch, torch.nn as nn

X = torch.rand(1000, 2)                 # модельные данные:
Y = X[:, 0] + 2 * X[:, 1] + 1  +  0.1 * torch.randn(X.size(0))
Y = Y.view(-1,1)

data_trn = Data((X[:800], Y[:800]), batch_size=100, shuffle=True)
data_val = Data((X[800:], Y[800:]), batch_size=200)

class Model(nn.Module):
    def __init__(self, *arg, **kvargs):        
        super().__init__()       

        self.cfg = Config(              # свойства архитектуры модели
            input  = 2,                 # размерность входа
            hidden = 10,                # число нейронов в скрытом слое
            output = 1,                 # размерность выхода
        )
        cfg = self.cfg(*arg, **kvargs)  # изменение параметров через конструктор

        self.mlp = MLP(input=cfg.input, hidden=cfg.hidden, output=cfg.output)
    
    def forward(self,x):                
        return self.mlp(x)

    def training_step(self, batch, batch_id):    
        x, y_true = batch   
        y_pred = self(x)                
        return {'loss': nn.functional.mse_loss(y_pred, y_true) }            

model = Model(hidden = 5)               # меняем скрытый слой

state = ModelState(model)               # выводим архитектуру
state.layers(input_size=(1, 2))
```

Мы получаем следеющую архитектуру (параметр `hidden` был изменён через конструктор):
```
Model                      params           data
├─ MLP                                       (1, 2) -> (1, 1)  <  mlp
│  └─ Sequential                             (1, 2) -> (1, 1)  <  mlp.layers
│     └─ Linear(2->5)          15  ~  71% |  (1, 2) -> (1, 5)  <  mlp.layers[0]
│     └─ GELU                                (1, 5) -> (1, 5)  <  mlp.layers[1]
│     └─ Dropout(0)                          (1, 5) -> (1, 5)  <  mlp.layers[2]
│     └─ Linear(5->1)           6  ~  29% |  (1, 5) -> (1, 1)  <  mlp.layers[3]
=================================
trainable:                     21
```

Теперь запускаем обучение, указывая, что нужно сохранять лучший валидационный loss:
```python
trainer = Trainer(model, data_trn, data_val)
trainer.set_optimizer( torch.optim.SGD(model.parameters(), lr = 0.1))

trainer.fit(epochs=10, monitor=['loss'])  # в папку "models/loss" будем  сохранять модели
```

Загрузка модели, вывод графика и продолжение обучение делается следующим образом (имя файла формируется тренером и в Вашем случае будет другим!):
```python
trainer = Trainer.load("models/loss/epoch_0009_loss_0.0098_07.01_11-02-17.pt", Model)

trainer.plot()                            # выводим график предыдущего обучения
trainer.data(trn=data_trn, val=data_val)  # задаём данные для тренировки и обучения
trainer.fit(epochs=50)                    # продолжаем обучение
```

Обратите внимание что, новый, созданный методом `load`, тренер не знает о данных.
Поэтому их необходимо передать тренеру через свойство `data`.
