# [QuNet](README.md) - Trainer.plot()

## Visualization of the training process

График с кривой обучения текущей модели  можно нарисовать при помощи метода тренера plot:
```python
trainer.plot()
```
Можно загрузить из файла модели только историю обучения и нарисовать её:
```python
hist = Trainer.load_hist("model.pt")
trainer.plot(hist)
```

Когда метод  `fit` получает аргумент `period_plot > 0`, тогда каждую `period_plot` эпоху выводится график с кривыми обучения. 
По умолчанию это две кривые: слева график loss, справа - score.
```python
trainer.fit(epochs=1000, period_plot=50)
```
По умолчанию, `period_plot=0` и график строится не будет. 

## Статистика обучения

После графика на экран выводится статистика обучения в следующем виде:
```
val_loss:  best = 0.691530[269], smooth21 = 0.695548[269], last21 = 0.701886 ± 0.007238
trn_loss:  best = 0.723788[259], smooth21 = 0.738220[269], last21 = 0.745274 ± 0.014665

val_score: best = 0.925800[264], smooth21 = 0.922200[269], last21 = 0.920948 ± 0.000000
trn_score: best = 0.905120[251], smooth21 = 0.898544[269], last21 = 0.895438 ± 0.002891

epochs=270, samples=13500000, steps=135000
times=(trn:222.55, val:10.49)m,  49.46 s/epoch, 98.91 s/10^3 steps,  989.10 s/10^6 samples
```
Первые две строки содержит информацию о loss, вторые две - о score.
Лучшее значение `best` в квадратных скобках содержит номер эпохи на которой оно было получено.
Аналогично для `smooth21` - это лучшее значение на кривой усреднённой по 21 точкам (см. свойства `trainer.view.smooth`).
Например, в случае переобучения, валидационный loss дстигает минимума и начинает увеличиваться.
Этот минимум можно определить не по абослютному лучшему выбросу `best`, а по сглаженной кривой.
Значение `last21` означает усреднение по 21 точке последних значений метрик.
В частности, если для loss значение `last21` существенно больше `smooth21`, это говорит об переобучении модели.

Далее указывается сколько прошло эпох, в тренировочном режиме показано примеров и сделано шагов оптимизатором.
Последняя строка - это общее время обучения и валидации, среднее время а секундах на эпоху, о 1000 шагов и 1000000 примеров.

Стаистики обучения вывести в любой момент, при помощи метода `stat`:
```python
trainer.stat()
```

## Внешний вид графиков

Все параметры визуализации в структурированном виде находятся в конфиге `trainer.view`. 
Их полный список можно найти в последнем разделе или исходниках (trainer.py).
Мы приведем примеры некоторых, наиболее часто встречаемых изменений параметров, принятых по умолчанию:

Обычно имеет смысл зафиксировать изменение метрик по оси y.
Тогда сразу видна разница в поведение кривых обучения, проведенных в различных экспериментах:
```python
trainer.view.loss (y_min=0.0, y_max=1.0, ticks=11)
trainer.view.score(y_min=0.5, y_max=1.0, ticks=6)
```
Необязательных параметр `ticks` задаёт число пометок со значенеями на оси.
Например, в первом случае 11 мететок в интервале [0...1] дадут равномерные значения 0.0, 0.1, ..., 1.0

Не показывать на графике ошибки learning rate:
```python
trainer.view.loss.lr = False   
```

<hr>

## Пометки

На графиках можно ставить пометки в виде верикальных пункирных линий с надписью возле них.
Такие пометки удобны для отражения на графике обучения некотрых радикальных изменений (динамически изменили архитектуру, заморозили часть сети и т.п.).
Такие пометки в историю добавляются методом `add_label` который вызывается перед запуском обучения `fit`:
```python
traner.add_label("frozen")
trainer.fit(epochs=10)
```

<hr>

## Сравнение нескольких моделей

<hr>

## Все параметры визуализации

You can customize the appearance of graphs using the following trainer options:

```python
trainer.view = Config(
    w  = 12,                   # plt-plot width
    h  =  5,                   # plt-plot height

    units = Config(            # 
        unit  = 'epoch',       # 'epoch' | 'sample'
        count = 1e6,           # units for number of samples
        time  = 's'            # time units: ms, s, m, h
    ),

    x_min = 0,                 # minimum value in samples on the x-axis (if < 0 last x_min samples)
    x_max = None,              # maximum value in samples on the x-axis (if None - last)

    smooth = Config(
        count  = 100,          # if the number of points exceeds count - draw a smooth line
        win    = 21,           # averaging window
        power   = 3,           # polynomial degree
        width  = 1.5,          # line thickness
        alpha  = 0.5,          # source data transparency
    ),
    loss = Config(
        show  = True,          # show loss subplot
        y_min = None,          # fixing the minimum value on the y-axis
        y_max = None,          # fixing the maximum value on the y-axis
        ticks = None,          # how many labels on the y-axis
        lr    = True,          # show learning rate
        labels= True,          # show labels (training events)
        trn_checks = False,    # show the achievement of the minimum training loss (dots)
        val_checks = True,     # show the achievement of the minimum validation loss (dots)
        last_checks = 100,     # how many last best points to display (if -1 then all)
        cfg   =  Config(),     # config to be displayed on the chart
        exclude = [],          # which config fields should be excluded
        fontsize = 8,          # font size for config output
    ),
    score = Config(
        show  = True,          # show score subplot
        y_min = None,          # fixing the minimum value on the y-axis
        y_max = None,          # fixing the maximum value on the y-axis
        ticks = None,          # how many labels on the y-axis
        lr    = True,          # show learning rate
        labels = True,         # show labels (training events)
        trn_checks = False,    # show the achievement of the optimum training score (dots)
        val_checks = True,     # show the achievement of the optimum validation score (dots)
        last_checks = 100,     # how many last best points to display (if -1 then all)
        cfg =  Config(),       # config to be displayed on the chart
        exclude = [],          # which config fields should be excluded
        fontsize = 8,          # font size for config output
    ),
)
```

<hr>
