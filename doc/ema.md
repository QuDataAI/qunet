# [QuNet](README.md) - EMA (Exponential Moving Average)

При обучении модели иногда полезно вычислять скользящие средние обученных параметров. Такая модель может дать лучший результат, так как менее подвержена переобучению.

Для включения EMA тренеру необходимо передать следующую конфигурацию:

```python
ema_cfg = Config(
    decay       = 0.99,  # коэффициент экспоненциального сглаживания
    start_epoch = 1,     # начиная с какой эпохи выполнять сглаживание
)

trainer = Trainer(model,
                  data_trn,
                  data_val,
                  ema_cfg=ema_cfg)
```

В процессе обучения модель со сглаженными весами будет сохранятся в параметре `model_ema` объекта класса `Trainer`

Если необходимо сохранять лучшие чекпоинты, добавляем в монитор метрики ['loss_ema', 'score_ema'] и соответствующие пути на диске:

```python

trainer.fit(epochs=100, monitor=['loss', 'score', 'loss_ema', 'score_ema', 'point'])

trainer.folders(loss_ema=f'models/loss_ema', score=f'models/score_ema')
```
