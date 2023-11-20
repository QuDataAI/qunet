import os, glob, math, copy, time, datetime
from pathlib import Path
from   tqdm.auto import tqdm
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch, torch.nn as nn

from .config   import Config
from .ema      import ModelEma
from .batch    import Batch
from .optim.scheduler    import Scheduler
from .plots    import plot_history

class Trainer:
    """
    Generic model training class.
    Any method can, of course, be overridden by inheritance or by an instance.
    """
    def __init__(self, model=None, data_trn=None, data_val=None, callbacks=[], score_max=True, wandb_cfg=None, ema_cfg=None) -> None:
        """
        Trainer

        Args:
        ------------
            model   (nn.Module):
                model for traininig; must have a training_step method
            data_trn (Data or DataLoader):
                training data
            data_val  (Data or DataLoader):
                data for validatio; may be missing;
            callbacks (Callback):
                list of Callback instances to be called on events
            score_max (bool):
                consider that the metric (the first column of the second tensor returned by the function `metrics` of the model ); should strive to become the maximum (for example, so for accuracy).
            wandb_cfg (Config):
                wandb params to external data tracking. You have to specify next:
                    api_key(str)
                        WANDB API key
                    project_name(str)
                        WANDB project name
                    run_name(str, optional)
                        WANDB run name
            ema_cfg (Config):
                EMA (Exponential Moving Average) support
                    decay(float)
                        average decay for EMA
                    start_epoch(int)
                        first epoch to average weights (avoid first random values impact)
        """
        self.model     = model
        self.score_max = score_max       # метрика должна быть максимальной (например accuracy)

        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype      = torch.float32 # see training large models;
        self.optim      = None

        self.schedulers = []             # список шедулеров для управления обучением
        self.scheduler  = Scheduler()    # текущий шедулер

        self.epoch      = 0              # текущая эпоха за всё время
        self.fit_epoch  = 0              # текущая эпоха после вызова fit

        self.callbacks  = callbacks      # list of Callback instances to be called on events
        self.wandb_cfg  = wandb_cfg      # wandb configuration

        self.data = Config(trn = data_trn,  val = data_val)

        self.ema_cfg = ema_cfg           # enable EMA (Exponential Moving Average) support
        self.model_ema = None            # reference to ema model

        self.best = Config(
            scores = {},                # best val scores
            loss = None,                # best val loss
            score_model = None,         # copy of the best model by val score
            loss_model  = None,         # copy of the best model by val loss
            score_ema_model = None,     # copy of the best EMA model by val score
            loss_ema_model  = None,     # copy of the best EMA model by val loss
            copy = False                # should copy loss or score model if is in monitor
        )

        self.folders = Config(
            loss   =    "models/loss",      # folder to save the best val loss models
            score  =    "models/score",     # folder to save the best val score models
            loss_ema =  "models/loss_ema",  # folder to save the best val loss EMA models
            score_ema = "models/score_ema", # folder to save the best val score EMA models
            point  =    "models/point",     # folder to save checkpoints
            prefix =    "",                 # add prefix to file name
        )

        self.folders.protect(False)    # for dynamic score folder creation

        # -------------------------------- настройки для построения графиков ошибок и метрик
        self.view = Config(
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

        self.view.protect(False)       # for dynamic score view creation

        # -------------------------------- история и текущие метрики процесса обучения модели
        self.hist = Config(            # история обучения и валидации:
            epochs  = 0,               # число эпох в режиме обучения (same self.epoch)
            samples = 0,               # число примеров в режиме обучения
            steps   = 0,               # число шагов градиентного спуска

            time = Config(
                trn  = 0,              # общее время тренировки
                val =  0               # общее время валидациии
            ),

            labels = [],
            trn  = Config(
                        best = Config(  loss = None,       # лучшее значение тренировочная ошибки
                                        scores = {},       # лучшие значения тенировочных метрик
                                        loss_samples  = 0, # когда была лучшая тренировочная ошибка
                                        loss_epochs   = 0, # когда была лучшая тренировочная ошибка
                                        score_samples = 0, # когда была лучшая тренировочная метрика
                                        score_epochs  = 0, # когда была лучшая тренировочная метрика
                                        losses=[],         # точки достижения лучшей ошибки  (loss,epochs,samples,steps)
                                      ),
                        epochs=[], samples=[], steps=[],   # история заначений после вызова training_step
                        batch_size=[], lr=[],
                        samples_epoch=[], steps_epoch=[],
                        times=[], losses=[], scores={}
                    ),
            val  = Config(
                        best = Config(  loss = None,       # лучшее значение валиадционная ошибки
                                        scores = {},       # лучшие значения тенировочных метрик
                                        loss_ema = None,   # лучшее значение валиадционная ошибки EMA модели
                                        scores_ema = {},   # лучшие значения тенировочных метрик (первой) EMA модели
                                        loss_samples  = 0, # когда была лучшая валиадционная ошибка
                                        loss_epochs   = 0, # когда была лучшая валиадционная ошибка
                                        losses=[],         # точки достижения лучшей ошибки  (loss,epochs,samples,steps)
                                    ),
                        epochs=[], samples=[], steps=[],    # история заначений после вызова validation_step
                        batch_size=[], lr=[], samples_epoch=[],
                        steps_epoch=[], times=[],
                        losses=[], scores={}
                    ),

            params = 0
        )

        self.compare = Config(
            hist = [],
            alpha= 0.3,
            trn  = True,
            val  = True,
        )

        if model is not None:
            self.hist.params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.wandb_cfg:
            self.wandb_init()
        else:
            self.wandb = None


    #---------------------------------------------------------------------------

    def add_label(self, text):
        """ Добавить пометку для графиков обучения """
        h = self.hist
        h.labels.append( [ text, h.epochs, h.samples, h.steps, h.time.trn, h.time.val ] )

    #---------------------------------------------------------------------------

    def set_optimizer(self, optim):
        """ Установить текущий оптимизатор """
        self.optim = optim
        self.set_optim_schedulers()

    def set_optim_schedulers(self):
        self.scheduler.optim = self.optim
        for sch in self.schedulers:
            sch.optim = self.optim

    #---------------------------------------------------------------------------

    def set_scheduler(self, scheduler):
        self.schedulers = []
        self.add_scheduler(scheduler)
        if scheduler.lr1 is not None:
            scheduler.set_lr(scheduler.lr1)

    def add_scheduler(self, scheduler):
        scheduler.optim = self.optim
        self.schedulers.append(scheduler)

    def del_scheduler(self, i):
        self.schedulers.pop(i)

    def clear_schedulers(self):
        self.scheduler = Scheduler(self.optim)
        self.schedulers = []

    def reset_schedulers(self):
        for sch in self.schedulers:
            sch.enable = True
            sch.done   = 0

    def stop_schedulers(self):
        for sch in self.schedulers:
            sch.enable = False

    def step_schedulers(self, epochs, samples):
        for sch in self.schedulers:
            if sch.enable:
                sch.step(epochs, samples)
                if not sch.enable:
                    self.add_label("")
                self.scheduler = sch
                break

    def plot_schedulers(self):
        self.scheduler.plot_list(self.schedulers)

    #---------------------------------------------------------------------------

    def to_device(self, batch):
        """ send mini-batch to device """
        if torch.is_tensor(batch) or Batch in batch.__class__.__bases__:
            return batch.to(self.device)
        if type(batch) is dict:
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)
            return batch
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = self.to_device(batch[i])
        return batch

    #---------------------------------------------------------------------------

    def get_fun_step(self, model, train):
        """ Получить функцию шага тренировки или валидации """
        fun_step = None
        if train:
            if hasattr(model, "training_step"):
                fun_step = model.training_step
        else:
            if hasattr(model, "validation_step"):
                fun_step = model.validation_step
            elif hasattr(model, "training_step"):
                fun_step = model.training_step

        assert fun_step is not None, "model must has training_step function"
        return fun_step

    #---------------------------------------------------------------------------

    def get_metrics(self, data):
        """ из результатов шага обучения выделить ошибку и метрики"""
        if torch.is_tensor(data):
            loss = data
            scores = None
        elif type(data) is dict:
            loss = data.get('loss')
            if loss is not None:
                del data['loss']
            if data:
                scores = data
            else:
                scores = None
        if scores:
            for score_name in scores.keys():
                if torch.is_tensor(scores[score_name]) and scores[score_name].ndim == 0:
                    scores[score_name] = scores[score_name].view(1)
        return loss, scores

    #---------------------------------------------------------------------------

    def samples_in_batch(self, batch):
        """ сколько примров в батче """
        if torch.is_tensor(batch):
            return len(batch)
        if type(batch) is list or type(batch) is tuple:
            return self.samples_in_batch(batch[0])
        if type(batch) is dict:
            return self.samples_in_batch(batch[list(batch.keys())[0]])
        if Batch in batch.__class__.__bases__:
            return len(batch)
        assert False, "wrong type of the fist element in batch"

    #---------------------------------------------------------------------------

    def fit_one_epoch(self, model, data,  train=True, accumulate_grad_batches=1, states=[], verbose=1):
        """
        Args
        ------------
        train (bool=True):
            режим тренировки, иначе валидации
        accumulate_grad_batches:
            аккумулировать градиент для accumulate батчей перед сдвигом оптимизатора; используем, когда большой батч или граф не помещаются в GPU
            https://kozodoi.me/blog/20210219/gradient-accumulation
        
        float16:
        https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
        """
        self.model.train(train)                     # режим обучение или тестирование
        torch.set_grad_enabled(train)               # строить или нет вычислительный граф

        scaler = None                               # используется для float16
        if torch.cuda.is_available() and self.dtype != torch.float32:
            scaler = torch.cuda.amp.GradScaler()

        if train:                                   # в режиме обучения
            self.hist.epochs  += 1                  # эпох за всё время (а если packs > 1 ?)
            self.epoch += 1                         # the same !!!
            self.optim.zero_grad()                  # обнуляем градиенты

        fun_step = self.get_fun_step(model, train)  # функция шага тренировки или валидации

        samples, steps, beg, lst = 0, 0, time.time(), time.time()
        counts_all, losses_all,  scores_all = torch.empty(0,1), None,  {}
        batch_id = 0  # может и раньше итератор прерваться (enumerate - плохо)
        epoch_scores = None
        for batch in data:
            num   = self.samples_in_batch(batch)

            if train:
                for callback in self.callbacks:
                    batch = callback.on_train_before_batch_transfer(self, self.model, batch, batch_id)

                batch = self.to_device(batch)

                for callback in self.callbacks:
                    batch = callback.on_train_after_batch_transfer(self, self.model, batch, batch_id)
            else:
                for callback in self.callbacks:
                    batch = callback.on_validation_before_batch_transfer(self, self.model, batch, batch_id)

                batch = self.to_device(batch)

                for callback in self.callbacks:
                    batch = callback.on_validation_after_batch_transfer(self, self.model, batch, batch_id)

            if scaler is None:
                step = fun_step(batch, batch_id)
            else:
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    step = fun_step(batch, batch_id)
            loss, step_scores = self.get_metrics(step)

            if train:
                if accumulate_grad_batches > 1:                  # normalize loss to account for batch accumulation (!)
                    loss = loss / accumulate_grad_batches

                if scaler is None:
                    loss.backward()                 # вычисляем градиенты
                else:
                    scaler.scale(loss).backward()   # вычисляем градиенты

                if accumulate_grad_batches > 1:                  # normalize loss to account for batch accumulation (!)
                    loss = loss * accumulate_grad_batches

                if (batch_id+1) % accumulate_grad_batches == 0 or (batch_id + 1 == len(data)):
                    if scaler is None:
                        self.optim.step()
                    else:
                        scaler.step(self.optim)     # подправляем параметры
                        scaler.update()             # Updates the scale for next iteration

                    for state in states:
                        state.update()

                    self.optim.zero_grad()          # обнуляем градиенты

                    steps      += 1                 # шагов за эпоху
                    self.hist.steps += 1            # шагов за всё время
                self.hist.samples += num            # примеров за всё время

                for callback in self.callbacks:
                    callback.on_after_step(self, self.model, batch, batch_id)                

            samples += num                          # просмотренных примеров за эпоху
            losses_all = loss.detach() if losses_all is None else torch.vstack([losses_all, loss.detach()])

            if step_scores is not None:
                for score_name in step_scores.keys():
                    score_step_data = step_scores[score_name].detach()
                    assert score_step_data.ndim == 1, f"scores should be averaged over the batch, but got shape:{score_step_data.shape}"
                    if score_name not in scores_all:
                        scores_all[score_name] = score_step_data
                    else:
                        scores_all[score_name] = torch.vstack([scores_all[score_name], score_step_data])

            counts_all = torch.vstack([counts_all, torch.Tensor([num])])

            if verbose and (time.time()-lst > 1 or batch_id+1 == len(data) ):
                lst = time.time()
                self.fit_progress(train, batch_id+1, len(data),
                                  losses_all, scores_all, counts_all, epoch_scores, samples, steps, time.time()-beg)

            if train:
                for callback in self.callbacks:
                    batch = callback.on_train_batch_end(self, self.model, batch, batch_id)
            else:
                for callback in self.callbacks:
                    batch = callback.on_validation_batch_end(self, self.model, batch, batch_id)
            
            batch_id += 1

        if hasattr(self.model, "on_epoch_end"):
            fun_on_epoch_end = model.on_epoch_end
            epoch_metrics = fun_on_epoch_end()
            _, epoch_scores = self.get_metrics(epoch_metrics)

        if train: self.hist.time.trn += (time.time()-beg)
        else:     self.hist.time.val += (time.time()-beg)

        if scores_all:
            for score_name in scores_all.keys():
                scores_all[score_name] = scores_all[score_name].cpu()
        else:
            scores_all = None

        if losses_all is not None:
            self.fit_progress(train, len(data), len(data), losses_all, scores_all, counts_all, epoch_scores, samples, steps, time.time()-beg)
            losses_all = losses_all.cpu()

        torch.set_grad_enabled(True)  
        return losses_all, scores_all, counts_all, epoch_scores, (samples, steps, time.time()-beg)

    #---------------------------------------------------------------------------

    def agg_metrics(self, steps_losses, steps_scores, steps_counts, epoch_scores):
        """ Вычислить среднее по всей эпохе """
        assert steps_losses is not None, "Do you calculate loss at all?"

        loss = ((steps_losses.detach().cpu() * steps_counts).sum(dim=0) / steps_counts.sum()).item()
        scores = {}
        if steps_scores is not None:
            for score_name in steps_scores.keys():
                scores[score_name] = ((steps_scores[score_name].detach().cpu() * steps_counts).sum(dim=0) / steps_counts.sum())[0]
        if epoch_scores is not None:
            for score_name in epoch_scores.keys():
                scores[score_name] = epoch_scores[score_name].detach().cpu()[0]
        if not scores:
            scores = None
        return (loss, scores)

    #---------------------------------------------------------------------------

    def fit_progress(self, train, batch_id, data_len, losses, scores, counts, epoch_scores, samples, steps, tm):
        """
        Вывод информации о прогрессе обучения (эпоха, время, ошибка и т.п.)
        В конкретном проекте можно перопределить.
        """
        loss, scores = self.agg_metrics(losses, scores, counts, epoch_scores)
        steps, samples = max(steps, 0), max(samples, 1)             # just in case
        st = ""
        if scores is not None:
            for score_name in scores.keys():
                st += f"{score_name}={scores[score_name]:.4f} "
        st += f"loss={loss:.4f};"
        st += " best"
        if scores is not None:
            for score_name in scores.keys():
                score_val = self.hist.val.best.scores.get(score_name).score if score_name in self.hist.val.best.scores else 0.0
                score_val_epoch = self.hist.val.best.scores.get(score_name).epochs if score_name in self.hist.val.best.scores else ' '
                st += f" {score_name}=(val:{score_val:.3f}[{score_val_epoch}]"
                if self.hist.val.best.scores_ema.get(score_name) and self.hist.val.best.scores_ema.get(score_name).score:
                    st += f" ema:{self.hist.val.best.scores_ema.get(score_name).score:.3f}"
                score_trn = self.hist.trn.best.scores.get(score_name).score if score_name in self.hist.trn.best.scores else 0.0
                score_trn_epoch = self.hist.trn.best.scores.get(score_name).epochs if score_name in self.hist.trn.best.scores else ' '
                st += f" trn:{score_trn:.3f}[{score_trn_epoch}]),"

        st += f" loss=(val:{(self.hist.val.best.loss or 0.0):.3f}[{self.hist.val.best.loss_epochs or ' '}]"
        if self.hist.val.best.loss_ema:
            st += f" ema:{(self.hist.val.best.loss_ema or 0.0):.3f}"
        st += f" trn:{(self.hist.trn.best.loss or 0.0):.3f}[{self.hist.trn.best.loss_epochs or ' '}])"

        print(f"\r{self.epoch:3d}{'t' if train else 'v'}[{batch_id:4d}/{data_len:4d}] {st} {data_len*tm/max(batch_id,1):.0f}s", end="   ")

    #---------------------------------------------------------------------------

    def unit_scales(self):
        """ Единицы измерения числа примеров и времени """
        t_unit = self.view.units.time  if  self.view.units.time in ['ms','s','m','h']  else 's'
        t_unit_scale = dict(ms=1e-3, s=1, m=60, h=3600)[t_unit]
        c_unit = self.view.units.count if self.view.units.count > 0  else 1
        c_unit_power = round(np.log10(c_unit), 0)
        return t_unit, t_unit_scale,  c_unit, c_unit_power

    #---------------------------------------------------------------------------

    def validate(self, model=None, data=None):
        """
        Validate model without gradient computation.
        The result is a dict with loss and score values.

        Args
        ------------
        model
            the model that makes the prediction (e.g. trainer.best.score_model)
        data
            dataset for prediction (its minibatch format should understand the model's predict_step method)
        """
        model = model or self.model
        data = data or self.data.val
        model.to(self.device)
        losses, scores, counts, epoch_scores, (samples_val, steps_val, tm_val) = self.fit_one_epoch(model, data, train=False)
        loss_val, score_val = self.agg_metrics(losses, scores, counts, epoch_scores)
        score_val = score_val.item() if len(score_val) == 1 else score_val
        return {'loss': loss_val, 'score': score_val }

    #---------------------------------------------------------------------------

    def predict(self, model, data, whole=False, batch_size=-1, n_batches=-1,  verbose:bool = True):
        """
        Calculate prediction for each example in data.
        The result is a dict whose composition depends on the dict returned by the model's predict_step method.

        Args
        ------------
        model
            the model that makes the prediction (e.g. trainer.best.score_model)
        data
            dataset for prediction (its minibatch format should understand the model's predict_step method)
        whole
            do not process fractional dataset batches
        batch_size
            minibatch size in examples (it will not change for dataset), if batch_size <= 0, then as in dataset
        n_batches
            number of minibatches (if n_batches n_batches <= 0, then all)
        verbose
            display information about the calculation process
        """
        model = model or self.model

        for callback in self.callbacks:
            callback.on_predict_start(self, model)

        model.to(self.device)
        model.train(False)               # режим тестирование
        torch.set_grad_enabled(False)    # вычислительный граф не строим
        data.whole = whole               # обычно по всем примерам (и по дробному батчу)

        assert hasattr(model, "predict_step"), "for prediction, the model needs to have method predict_step, witch return output tensor"

        if batch_size > 0:
            batch_size_save = data.batch_size
            data.batch_size = batch_size

        scaler = None
        if torch.cuda.is_available() and self.dtype != torch.float32:
            scaler = torch.cuda.amp.GradScaler()

        samples, beg, lst = 0, time.time(), time.time()
        res = dict()

        for batch_id, batch in enumerate(data):
            if n_batches > 0 and batch_id + 1 > n_batches:
                break

            for callback in self.callbacks:
                batch = callback.on_predict_before_batch_transfer(self, model, batch, batch_id)

            batch = self.to_device(batch)

            for callback in self.callbacks:
                batch = callback.on_predict_after_batch_transfer(self, model, batch, batch_id)


            if scaler is None:
                out = model.predict_step(batch, batch_id)
            else:
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    out = model.predict_step(batch, batch_id)

            if torch.is_tensor(out):
                out = {'output': out.detach()}

            for k,v in out.items():
                assert torch.is_tensor(v), "predict_step should return only tensor or dict of tensors"
                if k in res:
                    if v.ndim == 1:  res[k] = torch.vstack([res[k], v.view(-1,1).detach() ])
                    else:            res[k] = torch.vstack([res[k], v.detach() ])
                else:
                    if v.ndim == 1:  res[k] = v.view(-1,1).detach()
                    else:            res[k] = v.detach()

            if verbose and (time.time()-lst > 1 or batch_id+1 == len(data) ):
                lst = time.time()
                print(f"\r[{100*(batch_id+1)/len(data):3.0f}%]  {(time.time()-beg)/60:.2f}m", end=" ")


        if verbose:
            print(f" keys: {list(res.keys())}")
        if batch_size > 0:
            data.batch_size = batch_size_save
        for k in res.keys():
            res[k] = res[k].cpu()

        for callback in self.callbacks:
            callback.on_predict_end(self, model)

        torch.set_grad_enabled(True)
        return res

    #---------------------------------------------------------------------------

    def fit(self,  epochs=None,  samples=None, accumulate_grad_batches=1,
            period_plot:  int=0,  pre_val:bool=False,
            period_val:   int=1,  period_val_beg=1, val_beg:int = None,
            period_point: int=1,  point_start:int=1,

            monitor = [],
            patience = None,
            states = [],
            period_state: int=0,
            show_stat: bool = True,
            verbose=2):
        """
        Args
        ------------
            epochs (int):
                number of epochs for training (passes of one data_trn pack). If not defined (None) works "infinitely".
            samples (int):
                if defined, then will stop after this number of samples, even if epochs has not ended
            accumulate_grad_batches:
                аккумулировать градиент для accumulate батчей перед сдвигом оптимизатора; используем, когда большой батч или граф не помещаются в GPU
                https://kozodoi.me/blog/20210219/gradient-accumulation
            period_plot (int=0):
                period after which the training plot is displayed (in epochs)
            pre_val (bool=False):
                validate before starting training
            period_val (int=1):
                period after which validation run (in epochs)
            period_val_beg (int=1):
                validation period on the first val_beg epochs
            val_beg (int=None):
                the number of epochs from the start, after which the validation period will be equal to period_val.
            period_point (int=1):
                period after which checkpoints  are made(in epochs)
            point_start (int=1):
                period after fit runs will be saved checkpoints with period period_points (in epochs)
            monitor (list=[]):
                what to save in folders: monitor=['loss'] or monitor=['loss', 'score', 'point']
            patience (int):
                after how many epochs to stop if there was no better loss, but a better score during this time
            state (ModelState=None)
                an instance of the ModelState class that accumulates information about gradients on model parameters
            period_state (int=0)
                period after which the model state  plot is displayed (in epochs)
            show_stat (bool=True)
                show metrics after fit
            verbose (int=0,1,2)
                output of training information

        Example
        ```
        tariner = Trainer(model, data_trn=data_trn, data_val=data_val, score_max=True)
        trainer.view.units(count=1e6, time='m');
        trainer.best(copy=True)
        trainer.set_optimizer( torch.optim.Adam(model.parameters(), lr=1e-3) )
        trainer.fit(epochs=200, period_plot = 50, monitor=['score'], patience=100)
        ```
        """
        assert self.optim    is not None, "Define the optimizer first"
        assert self.data.trn is not None, "Define data.trn first"
        assert len(self.data.trn) > 0,    "You don't have a single training batch!"

        if type(states) not in [list, tuple]:
            states = [states]

        for callback in self.callbacks:
            callback.on_fit_start(self, self.model)

        self.set_optim_schedulers()
        self.model.to(self.device)

        if self.data.val is not None and hasattr(self.data.val, "reset"):
            self.data.val.reset()
        if hasattr(self.data.trn, "reset"):
            self.data.trn.reset()

        if pre_val and self.data.val is not None:
            assert len(self.data.val) > 0, "You don't have a single validation batch!"
            losses, scores, counts, epoch_scores, (samples_val, steps_val, tm_val) = self.fit_one_epoch(self.model, self.data.val, train=False)
            loss_val, score_val = self.agg_metrics(losses, scores, counts, epoch_scores)
            self.add_hist(self.hist.val, self.data.val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, self.scheduler.get_lr())
            print()

        epochs = epochs or 1_000_000_000
        best_epochs = [self.hist.val.best.loss_epochs]
        for score_name in self.hist.val.best.scores.keys():
            best_epochs.append(self.hist.val.best.scores[score_name].epochs)
        last_best = max(best_epochs)
        loss_val  = 0
        #for epoch in tqdm(range(1, epochs+1)):
        for epoch in range(1, epochs+1):
            self.fit_epoch = epoch               # epoch from start fit function
            for callback in self.callbacks:
                callback.on_epoch_start(self, self.model)
            for callback in self.callbacks:
                callback.on_train_epoch_start(self, self.model)

            losses, scores, counts, epoch_scores, (samples_trn,steps_trn,tm_trn) = self.fit_one_epoch(self.model,
                                                                                                      self.data.trn,
                                                                                                      train=True,
                                                                                                      accumulate_grad_batches=accumulate_grad_batches,
                                                                                                      states=states)
            loss_trn, score_trn = self.agg_metrics(losses, scores, counts, epoch_scores)
            lr = self.scheduler.get_lr()
            self.add_hist(self.hist.trn, self.data.trn.batch_size, samples_trn, steps_trn, tm_trn, loss_trn, score_trn, lr)

            if self.hist.trn.best.loss is None or self.hist.trn.best.loss > loss_trn:
                last_best = self.epoch
                self.hist.trn.best.loss = loss_trn
                self.hist.trn.best.loss_epochs  = self.hist.epochs
                self.hist.trn.best.loss_samples = self.hist.samples
                self.hist.trn.best.losses.append( (loss_trn, self.hist.epochs, self.hist.samples, self.hist.steps) )
                for callback in self.callbacks:
                    callback.on_best_loss(self, self.model)

            if score_trn:
                for score_name in score_trn.keys():
                    if score_name not in self.hist.trn.best.scores:
                        self.hist.trn.best.scores[score_name] = Config(score=None, epochs=0, samples=0, scores=[])
                    if self.best_score(self.hist.trn.best.scores[score_name].score, score_trn[score_name]):
                        last_best = self.epoch
                        self.hist.trn.best.scores[score_name].score = score_trn[score_name]
                        self.hist.trn.best.scores[score_name].epochs = self.hist.epochs
                        self.hist.trn.best.scores[score_name].samples = self.hist.samples
                        self.hist.trn.best.scores[score_name].scores.append((score_trn[score_name].item(), self.hist.epochs, self.hist.samples, self.hist.steps))
                        for callback in self.callbacks:
                            callback.on_best_score(self, self.model)

            self.update_ema_model(monitor)

            for callback in self.callbacks:
                callback.on_train_epoch_end(self, self.model)

            period = period_val_beg if val_beg and  self.epoch < val_beg  else period_val
            if  ( (period and self.epoch % period == 0) or epoch == epochs ):
                if self.data.val is not None:
                    for callback in self.callbacks:
                        callback.on_validation_epoch_start(self, self.model)

                    losses, scores, counts, epoch_scores, (samples_val,steps_val,tm_val) = self.fit_one_epoch(self.model, self.data.val, train=False)
                    loss_val, score_val = self.agg_metrics(losses, scores, counts, epoch_scores)
                    self.add_hist(self.hist.val, self.data.val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, lr)
                    self.ext_hist(loss_trn, score_trn, loss_val, score_val, lr)
                else:
                    loss_val  = loss_trn        # no validation data
                    score_val = score_trn

                # save best validation loss:
                if self.hist.val.best.loss is None or self.hist.val.best.loss > loss_val:
                    last_best = self.epoch
                    self.hist.val.best.loss =  self.best.loss = loss_val
                    self.hist.val.best.loss_epochs  = self.hist.epochs
                    self.hist.val.best.loss_samples = self.hist.samples
                    self.hist.val.best.losses.append((loss_val, self.hist.epochs, self.hist.samples, self.hist.steps))
                    if self.folders.loss and 'loss' in monitor:
                        self.save(Path(self.folders.loss) / Path(self.folders.prefix + f"epoch_{self.epoch:04d}_loss_{loss_val:.4f}_{self.now()}.pt"), model=self.model, optim=self.optim)
                    if self.best.copy and 'loss' in monitor:
                        self.best.loss_models  = copy.deepcopy(self.model)

                if score_val:
                    for score_name in score_val.keys():
                        if score_name not in self.hist.val.best.scores:
                            self.hist.val.best.scores[score_name] = Config(score=None, epochs=0, samples=0, scores=[])

                        if self.best_score(self.hist.val.best.scores[score_name].score, score_val[score_name]):
                            self.hist.val.best.scores[score_name].score = self.best.scores[score_name] = score_val[score_name]
                            self.hist.val.best.scores[score_name].epochs = self.hist.epochs
                            self.hist.val.best.scores[score_name].samples = self.hist.samples
                            self.hist.val.best.scores[score_name].scores.append(( score_val[score_name].item(), self.hist.epochs, self.hist.samples, self.hist.steps))
                            if self.folders.get(score_name) and score_name in monitor:
                                self.save(Path(self.folders.get(score_name)) / Path(self.folders.prefix + f"epoch_{self.epoch:04d}_score_{score_val[score_name]:.4f}_{self.now()}.pt"), model=self.model, optim=self.optim)
                            if self.best.copy and score_name in monitor:
                                self.best.score_model  = copy.deepcopy(self.model)

                for callback in self.callbacks:
                    callback.on_validation_epoch_end(self, self.model)

            if period_plot > 0 and (self.epoch % period_plot == 0 or epoch == epochs):
                if len(self.hist.trn.epochs) > 1:
                    self.plot()
                    for callback in self.callbacks:
                        callback.on_after_plot(self, self.model)
                if show_stat:
                    self.stat()
            if len(states) and period_state > 0 and  (self.epoch % period_state == 0 or epoch == epochs):
                for state in states:
                    state.plot()            

            if self.folders.point and 'point' in monitor and (period_point > 0 and point_start < self.epoch and self.epoch % period_point == 0 or epoch == epochs):
                for callback in self.callbacks:
                    callback.on_save_checkpoint(self, self.model, {})
                score_val = score_val if score_val is not None else {}
                score_trn = score_trn if score_trn is not None else {}
                fname = f"epoch_{self.epoch:04d}({self.now()})"
                for i, score_name in enumerate(score_val.keys()):
                    fname += f"_{score_name}(val_{score_val[score_name]:.4f}_trn_{score_trn[score_name]:.4f})"
                fname += f"_loss(val_{loss_val:.4f}_trn_{loss_trn:.4f}).pt"
                self.save(Path(self.folders.point) / Path(self.folders.prefix + fname), model=self.model, optim=self.optim)

            self.step_schedulers(1, samples_trn)

            if samples is not None:
                samples -= samples_trn
                if samples <= 0:
                    self.plot()
                    if show_stat:
                        self.stat()
                    break

            for callback in self.callbacks:
                callback.on_epoch_end(self, self.model)

            if patience is not None and patience > 0 and  self.epoch - last_best > patience:
                print(f"\n!!! Stop on patience={patience}. Epoch:{self.epoch}, last best score epoch:{self.hist.val.best.score_epochs}, best loss epoch:{self.hist.val.best.loss_epochs}")
                if period_plot > 0 and len(self.hist.trn.epochs) > 1:
                    self.plot()
                    for callback in self.callbacks:
                        callback.on_after_plot(self, self.model)
                    if show_stat:
                        self.stat()
                if period_state > 0:
                    for state in states:
                        state.plot()
                break

        if period_plot <= 0 and show_stat:            
            self.stat()

        for callback in self.callbacks:
            callback.on_fit_end(self, self.model)

        torch.set_grad_enabled(True) 
    #---------------------------------------------------------------------------

    def plot(self, view=None, hist=None, fname=None):
        """
        Plot training history
        """
        view = view or self.view
        hist = hist or self.hist
        plot_history(hist, view, fname, self.score_max, compare=self.compare)

    #---------------------------------------------------------------------------

    def stat(self, newline=True):
        if newline:
            print()

        trn_loss_bst,  trn_loss_epo,   val_loss_bst, val_loss_epo   = 0, 0, 0, 0
        trn_score_bst, trn_score_epo, val_score_bst, val_score_epo = 0, 0, 0, 0

        # best smoothing metrics
        win, power = self.view.smooth.win, self.view.smooth.power
        if len(self.hist.trn.losses) > 2*win:    
            smooth = savgol_filter(self.hist.trn.losses[win:], win, power)        
            smooth = np.hstack( [self.hist.trn.losses[:win], smooth] )
            trn_loss_epo = np.argmin(smooth)
            trn_loss_bst = smooth[trn_loss_epo]        
        if len(self.hist.val.losses) > 2*win:
            smooth = savgol_filter(self.hist.val.losses[win:], win, power)        
            smooth = np.hstack( [self.hist.val.losses[:win], smooth] )
            val_loss_epo = np.argmin(smooth)
            val_loss_bst = smooth[val_loss_epo]
        if len(self.hist.trn.scores) > 2*win:
            smooth = savgol_filter(self.hist.trn.scores[win:], win, power)  
            smooth = np.hstack( [self.hist.trn.scores[:win], smooth] )      
            trn_score_epo = np.argmax(smooth) if self.score_max else np.argmin(smooth)
            trn_score_bst = smooth[trn_score_epo]        
        if len(self.hist.val.scores) > 2*win:
            smooth = savgol_filter(self.hist.val.scores[win:], win, power)        
            smooth = np.hstack( [self.hist.val.scores[:win], smooth] )      
            val_score_epo = np.argmax(smooth) if self.score_max else np.argmin(smooth)
            val_score_bst = smooth[val_score_epo]

        # last 
        trn_loss_avr,  trn_loss_std,  val_loss_avr,  val_loss_std  = 0, 0, 0, 0
        trn_score_avr, trn_score_std, val_score_avr, val_score_std = 0, 0, 0, 0
        n = self.view.smooth.win
        if len(self.hist.trn.losses) > 1:
            trn_loss_avr = np.mean(self.hist.trn.losses[-n:]);  trn_loss_std = np.std(self.hist.trn.losses[-n:])
        if len(self.hist.val.losses) > 1:
            val_loss_avr = np.mean(self.hist.val.losses[-n:]);  val_loss_std = np.std(self.hist.val.losses[-n:])

        if self.data.val is not None and self.hist.val.best.loss is not None:                                                 
            print(f"val_loss:  best = {self.hist.val.best.loss:.6f}[{self.hist.val.best.loss_epochs or ' '}], smooth{n} = {val_loss_bst:.6f}[{val_loss_epo}], last{n} = {val_loss_avr:.6f} ± {val_loss_std:.6f}")                    
        if self.data.trn is not None and self.hist.trn.best.loss is not None:                                         
            print(f"trn_loss:  best = {self.hist.trn.best.loss:.6f}[{self.hist.trn.best.loss_epochs or ' '}], smooth{n} = {trn_loss_bst:.6f}[{trn_loss_epo}], last{n} = {trn_loss_avr:.6f} ± {trn_loss_std:.6f}")
        if self.data.val is not None and self.hist.val.best.scores:
            for score_name in self.hist.val.best.scores.keys():
                if len(self.hist.val.scores[score_name]) > 1:
                    val_score_avr = np.mean(self.hist.val.scores[score_name][-n:]); val_score_std = np.std(self.hist.val.scores[score_name][-n:])
                print(f"val_{score_name}: best = {self.hist.val.best.scores[score_name].score:.6f}[{self.hist.val.best.scores[score_name].epochs or ' '}], smooth{n} = {val_score_bst:.6f}[{val_score_epo}], last{n} = {val_score_avr:.6f} ± {val_score_std:.6f}")
        if self.data.trn is not None and self.hist.trn.best.scores:
            for score_name in self.hist.trn.best.scores.keys():
                if len(self.hist.trn.scores[score_name]) > 1:
                    trn_score_avr = np.mean(self.hist.trn.scores[score_name][-n:]); trn_score_std = np.std(self.hist.trn.scores[score_name][-n:])
                print(f"trn_{score_name}: best = {self.hist.trn.best.scores[score_name].score:.6f}[{self.hist.trn.best.scores[score_name].epochs or ' '}], smooth{n} = {trn_score_bst:.6f}[{trn_score_epo}], last{n} = {trn_score_avr:.6f} ± {trn_score_std:.6f}")
        if self.hist.val.best.loss_ema is not None:
            print(f"ema_val_loss={self.hist.val.best.loss_ema or 0.0:.6f}")
        if self.data.val is not None and self.hist.val.best.scores_ema:
            for score_name in self.hist.val.best.scores_ema.keys():
                if self.hist.val.best.scores_ema[score_name].score is not None:
                    print(f"ema_val_{score_name}: best = {self.hist.val.best.scores_ema[score_name].score:.6f}[{self.hist.val.best.scores_ema[score_name].epochs or ' '}]")

        print(f"epochs={self.epoch}, samples={self.hist.samples}, steps={self.hist.steps}")
        t_steps = f"{self.hist.time.trn*1_000/self.hist.steps:.2f}"   if self.hist.steps > 0 else "???"
        t_sampl = f"{self.hist.time.trn*1_000_000/self.hist.samples:.2f}" if self.hist.samples > 0 else "???"
        t_epoch = f"{self.hist.time.trn/self.epoch:.2f}" if self.epoch > 0 else "???"    
        print(f"times=(trn:{self.hist.time.trn/60:.2f}, val:{self.hist.time.val/60:.2f})m,  {t_epoch} s/epoch, {t_steps} s/10^3 steps,  {t_sampl} s/10^6 samples")

    #---------------------------------------------------------------------------

    def best_score(self, best, score):
        return score is not None \
                and (best is None \
                or (best < score and     self.score_max) \
                or (best > score and not self.score_max) )

    #---------------------------------------------------------------------------

    def now(self):
        return datetime.datetime.now().strftime("%m.%d_%H-%M-%S")

    #---------------------------------------------------------------------------

    def add_hist(self, hist, batch_size, samples, steps, tm, loss, scores, lr):
        hist.epochs       .append(self.hist.epochs)
        hist.samples      .append(self.hist.samples)
        hist.steps        .append(self.hist.steps)
        hist.samples_epoch.append(samples)
        hist.steps_epoch  .append(steps)
        hist.batch_size   .append(batch_size)
        hist.times        .append(tm)
        hist.lr           .append(lr)
        hist.losses       .append(loss)
        if scores is not None:
            for score_name in scores.keys():
                if score_name not in hist.scores:
                    hist.scores[score_name] = []
                hist.scores[score_name].append(scores[score_name].item())

    #---------------------------------------------------------------------------

    def ext_hist(self, loss_trn, score_trn, loss_val, score_val, lr):
        if self.wandb:
            ext_data = {'loss_trn': loss_trn, 'loss_val': loss_val, 'lr': lr}
            if score_trn:
                for score_param in score_trn.keys():
                    ext_data[f"{score_param}_trn"] = score_trn[score_param]
            if score_val:
                for score_param in score_val.keys():
                    ext_data[f"{score_param}_val"] = score_val[score_param]
            self.wandb.log(ext_data)

    #---------------------------------------------------------------------------

    def save(self, fname, model = None, optim=None, info=""):
        try:
            model = model or self.model
            assert model is not None, "You don't have a model!"
            optim = optim or self.optim

            cfg = model.cfg if hasattr(model, "cfg") else Config()
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            state = {
                'format':          1,                         # версия формата сейва
                'info':            info,
                'date':            datetime.datetime.now(),   # дата и время
                'model' :          model.state_dict(),        # параметры модели
                'config':          cfg,                       # конфигурация модели
                'class':           model.__class__,
                'hist':            self.hist,
                'view':            self.view,

                'optim':           optim.__class__ if optim is not None else None,
                'optim_cfg':       None,
                'optim_state':     None,
            }
            if optim is not None:
                if hasattr(optim, "defaults"):
                    state['optim_cfg'] = optim.defaults
                elif hasattr(optim, "cfg"):
                    state['optim_cfg'] = optim.cfg

                if hasattr(optim, "state_dict"):
                    state['optim_state'] = optim.state_dict()
            torch.save(state, fname)
        except:
            print(f"Something  wrong in the function trainer.save fname: '{fname}'")

    #---------------------------------------------------------------------------

    def resume(self, fname=None):
        if fname is None:
            list_of_points = []
            for folder in [self.folders.loss, self.folders.score, self.folders.point]:
                list_of_points.extend(glob.glob(f'{folder}/*') if folder else [])
            assert len(list_of_points) != 0, 'no saves for resume'
            fname = max(list_of_points, key=os.path.getctime)
        
        print('resume from weights', fname)
        Trainer.load(fname, None, self)

    #---------------------------------------------------------------------------

    def load(fname, ClassModel=None, trainer=None):
        """
        Static method of created trainer with loaded model.

        Args
        ------------
        fname (str):
            file name with model parameters
        ClassModel:
            name of model class

        Example
        ------------
        class Model(nn.Module):
            ...

        trainer.save("cur_model.pt")
        ...

        new_trainer = Trainer.load("cur_model.pt", Model)
        new_trainer.plot()
        """ 

        #try:
        state = torch.load(fname)  # , map_location='cpu'
        state = Trainer.backward_compatibility(state)

        print(f"info:  {state.get('info', '???')}")
        print(f"date:  {state.get('date', '???')}")
        print(f"model: {state.get('class','???')}")
        print(f"optim: {state.get('optim','???')}")
        #print(f"       {state.get('optim_cfg','???')}")
        #except:
        #    print(f"Cannot open file: '{fname}'")
        #    return None

        assert type(state) == dict,  f"Apparently this model was not saved by the trainer: state:{type(state)}"
        assert 'model'  in state,    f"Apparently this model was not saved by the trainer: no 'model' in state: {list(state.keys())}"
        assert 'config' in state,    f"Apparently this model was not saved by the trainer: no 'config' in state: {list(state.keys())}"

        trainer = trainer or Trainer(None, None)
        
        if ClassModel is None:
            ClassModel = state.get('class')
            
        if len(state['config'].get_str()) == 0:
            trainer.model = ClassModel()            
        else:
            trainer.model = ClassModel(state['config'])            
            
        trainer.model.to(trainer.device)
            
        if trainer.model is not None:        
            trainer.model.load_state_dict(state['model'])            

        if trainer.model is not None:
            if 'optim_state' in state and 'optim' in state:
                if state['optim'] == torch.optim.SGD:
                    trainer.optim  = torch.optim.SGD(trainer.model.parameters(), lr=1e-5)
                elif state['optim'] == torch.optim.Adam:
                    trainer.optim  = torch.optim.Adam(trainer.model.parameters(), lr=1e-5)
                elif state['optim'] == torch.optim.AdamW:
                    trainer.optim  = torch.optim.AdamW(trainer.model.parameters(), lr=1e-5)

            if trainer.optim is not None:
                try:
                    if 'optim_state' in state and state['optim_state'] is not None:
                        trainer.optim.load_state_dict(state['optim_state'])
                    elif 'state_dict' in state and state['state_dict'] is not None:  # old version  
                        trainer.optim.load_state_dict(state['state_dict'])
                except:
                    print("!!!! Trainer.load> can't trainer.optim.load_state_dict")

        trainer.hist(state['hist'])
        trainer.view(state['view'])
        trainer.epoch = trainer.hist.epochs
        print(f"epoch: {trainer.epoch}")        
        return trainer

    #---------------------------------------------------------------------------

    def load_hist(fname, verbose=0):
        """
        Load training history from file fname

        Args
        ------------
        fname (str):
            file name with model parameters

        Example
        ------------
        class Model(nn.Module):
            ...

        hist = trainer.load_hist("cur_model.pt")
        trainer.plot(hist)
        """
        try:
            state = torch.load(fname, map_location='cpu')
            state = Trainer.backward_compatibility(state)
            if verbose:
                print(f"info:  {state.get('info', '???')}")
                print(f"date:  {state.get('date', '???')}")
                print(f"model: {state.get('class','???')}")
                print(f"optim: {state.get('optim','???')}")
                #print(f"       {state.get('optim_cfg','???')}")
        except:
            print(f"Cannot open file: '{fname}'")
            return None

        return state.get('hist')
    
    #---------------------------------------------------------------------------

    def transfer(self, fname):
        """ Load to currnet model parameters from file fname with another model """                
        state = torch.load(fname)
        self.model.load_state_dict(state['model'])        

    #---------------------------------------------------------------------------
    @staticmethod
    def backward_compatibility(state):
        """ Поддержка предыдущих форматов """
        if state.get('format', 0) < 1:
            state = Trainer.backward_compatibility_0_1(state)
        return state

    @staticmethod
    def backward_compatibility_0_1(state):
        """ Преобразование 0-й версии в 1-ю """
        if state['hist'].has('trn'):
            if state['hist'].trn.has('scores'):
                state['hist'].trn.scores = {'score': state['hist'].trn.scores}
            if state['hist'].trn.has('best'):
                state['hist'].trn.best.scores = {'score': Config(score=state['hist'].trn.best.score,
                                                              epochs=state['hist'].trn.best.score_epochs,
                                                              samples=state['hist'].trn.best.score_samples,
                                                              scores=state['hist'].trn.best.scores)}
        if state['hist'].has('val'):
            if state['hist'].val.has('scores'):
                state['hist'].val.scores = {'score': state['hist'].val.scores}
            if state['hist'].val.has('best'):
                state['hist'].val.best.scores = {'score': Config(score=state['hist'].val.best.score,
                                                              epochs=state['hist'].val.best.score_epochs,
                                                              samples=state['hist'].val.best.score_samples,
                                                              scores=state['hist'].val.best.scores)}
        state['format'] = 1
        return state

    def wandb_init(self):
        """
        """
        import wandb

        assert hasattr(self.wandb_cfg, "api_key"), "Define api_key param for login to WANDB"
        assert hasattr(self.wandb_cfg, "project_name"), "Define project_name param for attach current run to it"

        run_name = self.wandb_cfg.run_name if hasattr(self.wandb_cfg, "run_name") else None

        cfg = self.model.cfg if hasattr(self.model, "cfg") else Config()

        self.wandb = wandb

        self.wandb.login(key=self.wandb_cfg.api_key)
        self.wandb.init(
            # set the wandb project where this run will be logged
            project=self.wandb_cfg.project_name,
            name=run_name,
            config=cfg,
            #resume="must"
        )

    #---------------------------------------------------------------------------

    def update_ema_model(self, monitor):
        """
        Update weigths of EMA model with: decay * model + (1. - decay) * model_ema
        """
        if self.data.val is None:
            return
        
        # create EMA model if not exists
        if self.ema_cfg and (self.model_ema is None) and (self.ema_cfg.start_epoch < self.epoch):
            self.model_ema = ModelEma(self.model, self.ema_cfg.decay, self.device)
            return

        if self.model_ema is None:
            return

        # update weights
        self.model_ema.update(self.model)

        # validate model
        
        losses, scores, counts, epoch_scores, (samples_val, steps_val, tm_val) = self.fit_one_epoch(self.model_ema.module, self.data.val, train=False)
        loss_val_ema, score_val_ema = self.agg_metrics(losses, scores, counts, epoch_scores)

        # save best EMA validation loss:
        if self.hist.val.best.loss_ema is None or self.hist.val.best.loss_ema > loss_val_ema:
            self.hist.val.best.loss_ema = loss_val_ema
            if self.folders.loss_ema and 'loss_ema' in monitor:
                self.save(Path(self.folders.loss_ema) / Path(
                    self.folders.prefix + f"epoch_{self.epoch:04d}_loss_{loss_val_ema:.4f}_{self.now()}.pt"), model=self.model_ema.module)
            if self.best.copy and 'loss_ema' in monitor:
                self.best.loss_ema_model = copy.deepcopy(self.model_ema.module)

        # save best EMA validation score:
        if score_val_ema:
            for score_name in score_val_ema.keys():
                if score_name not in self.hist.val.best.scores_ema:
                    self.hist.val.best.scores_ema[score_name] = Config(score=None, epochs=0, samples=0, scores=[])

                if self.best_score(self.hist.val.best.scores_ema[score_name].score, score_val_ema[score_name]):
                    self.hist.val.best.scores_ema[score_name].score = score_val_ema[score_name]
                    if self.folders.get(f'{score_name}_ema') and f'{score_name}_ema' in monitor:
                        self.save(Path(self.folders.get(f'{score_name}_ema')) / Path(self.folders.prefix + f"epoch_{self.epoch:04d}_score_{score_val_ema[score_name]:.4f}_{self.now()}.pt"), model=self.model_ema.module)
                    if self.best.copy and f'{score_name}_ema' in monitor:
                        self.best.score_ema_model = copy.deepcopy(self.model_ema.module)
