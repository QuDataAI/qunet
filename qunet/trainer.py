import os, math, copy, time, datetime
from pathlib import Path
from   tqdm.auto import tqdm
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from .utils    import Config
from .callback import Callback
from .batch    import Batch
from .optim    import Scheduler
from .plots    import plot_history

class Trainer:
    """
    Generic model training class.
    Any method can, of course, be overridden by inheritance or by an instance.
    """
    def __init__(self, model=None, data_trn=None, data_val=None, callbacks=[], score_max=True) -> None:
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
        """
        self.model     = model
        self.score_max = score_max       # метрика должна быть максимальной (например accuracy)

        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype      = torch.float32 # see training large models;
        self.optim      = None
        self.schedulers = []             # список шедулеров для управления обучением
        self.scheduler  = Scheduler()    # текущий шедулер

        self.epoch      = 0              # текущая эпоха после вызова fit
        self.callbacks  = callbacks      # list of Callback instances to be called on events

        self.data = Config(trn = data_trn,  val = data_val)

        self.best = Config(
            score = None,               # best val score
            loss = None,                # best val loss
            score_model = None,         # copy of the best model by val score
            loss_model  = None,         # copy of the best model by val loss
            copy = False                 # should copy loss or score model if is in monitor
        )

        self.folders = Config(
            loss   = None,              # folder to save the best val loss models
            score  = None,              # folder to save the best val score models
            points = None               # folder to save checkpoints
        )

        # -------------------------------- настройки для построения графиков ошибок и метрик
        self.view = Config(
            w  = 12,                   # plt-plot width
            h  =  5,                   # plt-plot height
            units = Config(
                unit  = 'epoch',       # 'epoch' | 'sample'
                count = 1e6,           # units for number of samples
                time  = 's'            # time units: ms, s, m, h
            ),

            x_min = 0,                 # minimum value in samples on the x-axis (if < 0 last x_min samples)
            x_max = None,              # maximum value in samples on the x-axis (if None - last)

            loss = Config(
                show  = True,          # show loss subplot
                y_min = None,          # fixing the minimum value on the y-axis
                y_max = None,          # fixing the maximum value on the y-axis
                ticks = None,          # how many labels on the y-axis
                lr    = True,          # show learning rate
                labels= True,          # show labels (training events)
                trn_checks = False,     # show the achievement of the minimum training loss (dots)
                val_checks = True,     # show the achievement of the minimum validation loss (dots)
                last_checks = 100          # how many last best points to display (if -1 then all)
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
                last_checks = 100      # how many last best points to display (if -1 then all)
            ),
        )

        # -------------------------------- история и текущие метрики процесса обучения модели
        self.hist = Config(            # история обучения и валидации:
            epochs  = 0,               # число эпох в режиме обучения
            samples = 0,               # число примеров в режиме обучения
            steps   = 0,                # число шагов градиентного спуска

            time = Config(
                trn  = 0,              # общее время тренировки
                val =  0               # общее время валидациии
            ),

            labels = [],
            trn  = Config(
                        best = Config(  loss = None,       # лучшее значение тренировочная ошибки
                                        score = None,      # лучшее значение тенировочной метрики (первой)
                                        loss_samples  = 0, # когда была лучшая тренировочная ошибка
                                        loss_epochs   = 0, # когда была лучшая тренировочная ошибка
                                        score_samples = 0, # когда была лучшая тренировочная метрика
                                        score_epochs  = 0, # когда была лучшая тренировочная метрика
                                        losses=[],         # точки достижения лучшей ошибки  (loss,epochs,samples,steps)
                                        scores=[]          # точки достижения лучшей метрики (score,epochs,samples,steps)
                                      ),
                        epochs=[], samples=[], steps=[],   # история заначений после вызова training_step
                        batch_size=[], lr=[],
                        samples_epoch=[], steps_epoch=[],
                        times=[], losses=[], scores=[]
                    ),
            val  = Config(
                        best = Config(  loss = None,       # лучшее значение валиадционная ошибки
                                        score = None,      # лучшее значение валиадционная метрики (первой)
                                        loss_samples  = 0, # когда была лучшая валиадционная ошибка
                                        loss_epochs   = 0, # когда была лучшая валиадционная ошибка
                                        score_samples = 0, # когда была лучшая валиадционная метрика
                                        score_epochs  = 0, # когда была лучшая валиадционная метрика
                                        losses=[],         # точки достижения лучшей ошибки  (loss,epochs,samples,steps)
                                        scores=[]          # точки достижения лучшей метрики (score,epochs,samples,steps)
                                    ),
                        epochs=[], samples=[], steps=[],    # история заначений после вызова validation_step
                        batch_size=[], lr=[], samples_epoch=[],
                        steps_epoch=[], times=[],
                        losses=[], scores=[]
                    ),

            params = 0
        )
        if model is not None:
            self.hist.params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    def get_metrics(self, step):
        """ из результатов шага обучения вделить ошибку и метрику"""
        if torch.is_tensor(step):
            loss   = step
            scores = None
        elif type(step) is dict:
            loss  = step.get('loss')
            scores = step.get('score')
        if torch.is_tensor(scores) and scores.ndim == 0:
            scores = scores.view(1)
        return loss, scores

    #---------------------------------------------------------------------------

    def samples_in_batch(self, batch):
        """ сколько примров в батче """
        if torch.is_tensor(batch):
            return len(batch)
        if type(batch) is list or type(batch) is tuple:
            return self.samples_in_batch(batch[0])
        if Batch in batch.__class__.__bases__:
            return len(batch)
        assert False, "wrong type of the fist element in batch"

    #---------------------------------------------------------------------------

    def fit_epoch(self, epoch, model, data,  train=True, accumulate=1, verbose=1):
        """
        Args:
            * train      - True: режим тренировки, иначе валидации
            * accumulate - аккумулировать градиент для стольки батчей перед сдвигом оптимизатора;
                           используем, когда большой батч или граф не помещаются в GPU
        https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
        """
        self.model.train(train)                     # режим обучение или тестирование
        torch.set_grad_enabled(train)               # строить или нет вычислительный граф

        scaler = None                               # используется для float16
        if torch.cuda.is_available() and self.dtype != torch.float32:
            scaler = torch.cuda.amp.GradScaler()

        if train:                                   # в режиме обучения
            self.hist.epochs  += 1                  # эпох за всё время (а если packs > 1 ?)
            self.optim.zero_grad()                  # обнуляем градиенты

        fun_step = self.get_fun_step(model, train)  # функция шага тренировки или валидации        

        samples, steps, beg, lst = 0, 0, time.time(), time.time()
        counts_all, losses_all,  scores_all = torch.empty(0,1), None,  None
        for batch_id, batch in enumerate(data):
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
            loss, scores = self.get_metrics(step)

            if train:
                if scaler is None:
                    loss.backward()   # вычисляем градиенты
                else:
                    scaler.scale(loss).backward()   # вычисляем градиенты
                if (batch_id+1) % accumulate == 0:
                    if scaler is None:
                        self.optim.step()
                    else:
                        scaler.step(self.optim)     # подправляем параметры
                        scaler.update()             # Updates the scale for next iteration
                    self.optim.zero_grad()          # обнуляем градиенты
                    steps      += 1                 # шагов за эпоху
                    self.hist.steps += 1            # шагов за всё время
                self.hist.samples += num            # примеров за всё время

            samples += num                          # просмотренных примеров за эпоху
            losses_all = loss.detach() if losses_all is None else torch.vstack([losses_all, loss.detach()])
            if scores is not None:
                scores = scores.detach()            # just in case
                assert scores.ndim == 1, f"scores should be averaged over the batch, but got shape:{scores.shape}"
                scores_all = scores if scores_all is None else torch.vstack([scores_all, scores])
            counts_all = torch.vstack([counts_all, torch.Tensor([num])])

            if verbose and (time.time()-lst > 1 or batch_id+1 == len(data) ):
                lst = time.time()
                self.fit_progress(epoch, train, batch_id+1, len(data),
                                  losses_all, scores_all, counts_all, samples, steps, time.time()-beg)

        if train: self.hist.time.trn += (time.time()-beg)
        else:     self.hist.time.val += (time.time()-beg)

        if scores_all is not None:
            scores_all = scores_all.cpu()
        if losses_all is not None:
            self.fit_progress(epoch, train, len(data), len(data), losses_all, scores_all, counts_all, samples, steps, time.time()-beg)
            losses_all = losses_all.cpu()

        return losses_all, scores_all, counts_all, (samples, steps, time.time()-beg)

    #---------------------------------------------------------------------------

    def mean(self, losses, scores, counts):
        """ Вычислить среднее по всей эпохе """
        loss  = ((losses.detach().cpu() * counts).sum(dim=0) / counts.sum()).item()
        if scores is not None:
            scores = ((scores.detach().cpu() * counts).sum(dim=0) / counts.sum())
        return (loss, scores)

    #---------------------------------------------------------------------------

    def fit_progress(self, epoch, train, batch_id, data_len, losses, scores, counts, samples, steps, tm):
        """
        Вывод информации о прогрессе обучения (эпоха, время, ошибка и т.п.)
        В конкретном проекте можно перопределить.
        """
        loss, score = self.mean(losses, scores, counts)
        steps, samples = max(steps, 0), max(samples, 1)             # just in case
        st = ""
        if score is not None and len(score):
            st += f"score={score[0]:.4f} "                          # главная метрика
            if len(score) > 1: st += "("                            # вспомогательные
            for i in range(1, len(score)):
                st += f"{score[i]:.4f}" + (", " if i+1 < len(score) else ") ")
        st += f"loss={loss:.4f};"
        st += f" best score=(val:{(self.hist.val.best.score or 0.0):.4f}[{self.hist.val.best.score_epochs or ' '}] trn:{(self.hist.trn.best.score or 0.0):.4f}[{self.hist.trn.best.score_epochs or ' '}]),"
        st += f" loss=(val:{(self.hist.val.best.loss or 0.0):.4f}[{self.hist.val.best.loss_epochs or ' '}] trn:{(self.hist.trn.best.loss or 0.0):.4f}[{self.hist.trn.best.loss_epochs or ' '}])"

        t_unit, t_unit_scale,  c_unit, c_unit_power = self.unit_scales()
        #data_len = str(data_len)
        #batch_id = str(batch_id).rjust(len(data_len))
        #print(f"\r{epoch:3d}{'t' if train else 'v'}[{batch_id}/{data_len}] {st}", end="   ")
        print(f"\r{self.hist.epochs:3d}{'t' if train else 'v'}[{batch_id:4d}/{data_len:4d}] {st}", end="   ")

    #---------------------------------------------------------------------------

    def unit_scales(self):
        """ Единицы измерения числа примеров и времени """
        t_unit = self.view.units.time  if  self.view.units.time in ['ms','s','m','h']  else 's'
        t_unit_scale = dict(ms=1e-3, s=1, m=60, h=3600)[t_unit]
        c_unit = self.view.units.count if self.view.units.count > 0  else 1
        c_unit_power = round(np.log10(c_unit), 0)
        return t_unit, t_unit_scale,  c_unit, c_unit_power

    #---------------------------------------------------------------------------

    def predict(self, model, data, whole=False, batch_size=-1, n_batches=-1,  verbose:bool = True):
        """
        Calculate prediction for each example in data.
        The result is a dict whose composition depends on the dict returned by the model's predict_step method.
        Args:
            * model - the model that makes the prediction (e.g. trainer.best.score_model)
            * data - dataset for prediction (its minibatch format should understand the model's predict_step method)
            * whole - do not process fractional dataset batches
            * batch_size - minibatch size in examples (it will not change for dataset), if batch_size <= 0, then as in dataset
            * n_batches - number of minibatches (if n_batches n_batches <= 0, then all)
            * verbose - display information about the calculation process
        """
        for callback in self.callbacks:
            callback.on_predict_start(self, self.model)

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
                batch = callback.on_predict_before_batch_transfer(self, self.model, batch, batch_id)

            batch = self.to_device(batch)

            for callback in self.callbacks:
                batch = callback.on_predict_after_batch_transfer(self, self.model, batch, batch_id)


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
            callback.on_predict_end(self, self.model)

        return res

    #---------------------------------------------------------------------------

    def fit(self,  epochs=None,  samples=None,
            pre_val:bool=False, period_val:int=1, period_plot:int=100,
            period_points:int=1, period_val_beg=1, samples_beg:int = None,            
            monitor = [],
            patience = None):        
        """
        Args
        ------------                
            epochs (int):
                number of epochs for training (passes of one data_trn pack). If not defined (None) works "infinitely".
            samples (int):
                if defined, then will stop after this number of samples, even if epochs has not ended
            pre_val (bool=False):
                validate before starting training
            period_val (int=1):
                period after which validation run (in epochs)
            period_plot (int=100):
                period after which the training plot is displayed (in epochs)
            period_val_beg (int=1):
                validation period on the first samples_beg examples
            samples_beg (int):
                the number of samples from the start, after which the validation period will be equal to period_val.
            period_points (int=1):
                period after which checkpoints are made and the current model is saved (in epochs)
            monitor (list=[]):
                what to save in folders: monitor=['loss'] or monitor=['loss', 'score', 'points']
            patience (int):
                after how many epochs to stop if there was no better loss, but a better score during this time

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

        for callback in self.callbacks:
            callback.on_fit_start(self, self.model)

        self.set_optim_schedulers()
        self.model.to(self.device)

        if self.data.val is not None and hasattr(self.data.val, "reset"):
            self.data.val.reset()
        if hasattr(self.data.trn, "reset"):
            self.data.trn.reset()

        self.epoch = 0
        if pre_val and self.data.val is not None:
            losses, scores, counts, (samples_val, steps_val, tm_val) = self.fit_epoch(0, self.model, self.data.val, train=False)
            loss_val, score_val = self.mean(losses, scores, counts)
            self.add_hist(self.hist.val, self.data.val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, self.scheduler.get_lr())
            print()

        epochs = epochs or 1_000_000_000
        last_best = 0
        #for epoch in tqdm(range(1, epochs+1)):        
        for epoch in range(1, epochs+1):
            self.epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_start(self, self.model)
            for callback in self.callbacks:
                callback.on_train_epoch_start(self, self.model)

            losses, scores, counts, (samples_trn,steps_trn,tm_trn) = self.fit_epoch(epoch, self.model, self.data.trn, train=True)
            loss_trn, score_trn = self.mean(losses, scores, counts)
            lr = self.scheduler.get_lr()
            self.add_hist(self.hist.trn, self.data.trn.batch_size, samples_trn, steps_trn, tm_trn, loss_trn, score_trn, lr)

            if self.hist.trn.best.loss is None or self.hist.trn.best.loss > loss_trn:
                last_best = epoch
                self.hist.trn.best.loss = loss_trn
                self.hist.trn.best.loss_epochs  = self.hist.epochs
                self.hist.trn.best.loss_samples = self.hist.samples
                self.hist.trn.best.losses.append( (loss_trn, self.hist.epochs, self.hist.samples, self.hist.steps) )
                for callback in self.callbacks:
                    callback.on_best_loss(self, self.model)


            if self.best_score(self.hist.trn.best.score, score_trn):
                last_best = epoch
                self.hist.trn.best.score = score_trn[0]
                self.hist.trn.best.score_epochs  = self.hist.epochs
                self.hist.trn.best.score_samples = self.hist.samples
                self.hist.trn.best.scores.append((score_trn[0].item(), self.hist.epochs, self.hist.samples, self.hist.steps))
                for callback in self.callbacks:
                    callback.on_best_score(self, self.model)

            for callback in self.callbacks:
                callback.on_train_epoch_end(self, self.model)

            period = period_val_beg if samples_beg and  self.hist['samples'] < samples_beg else period_val
            if  self.data.val is not None  and (period and epoch % period == 0) or epoch == epochs:
                for callback in self.callbacks:
                    callback.on_validation_epoch_start(self, self.model)

                losses, scores, counts, (samples_val,steps_val,tm_val) = self.fit_epoch(epoch, self.model, self.data.val, train=False)
                loss_val, score_val = self.mean(losses, scores, counts)
                self.add_hist(self.hist.val, self.data.val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, lr)

                # save best validation loss:
                if self.hist.val.best.loss is None or self.hist.val.best.loss > loss_val:
                    last_best = epoch
                    self.hist.val.best.loss =  self.best.loss = loss_val
                    self.hist.val.best.loss_epochs  = self.hist.epochs
                    self.hist.val.best.loss_samples = self.hist.samples
                    self.hist.val.best.losses.append((loss_val, self.hist.epochs, self.hist.samples, self.hist.steps))
                    if self.folders.loss and 'loss' in monitor:
                        self.save(Path(self.folders.loss) / Path(f"loss_{loss_val:.4f}_{self.now()}.pt"), model=self.model, optim=self.optim)
                    if self.best.copy and 'loss' in monitor:
                        self.best.loss_models  = copy.deepcopy(self.model)

                if self.best_score(self.hist.val.best.score, score_val):
                    self.hist.val.best.score = self.best.score = score_val[0]
                    self.hist.val.best.score_epochs  = self.hist.epochs
                    self.hist.val.best.score_samples = self.hist.samples
                    self.hist.val.best.scores.append( ( score_val[0].item(), self.hist.epochs, self.hist.samples, self.hist.steps) )
                    if self.folders.score and 'score' in monitor:
                        self.save(Path(self.folders.score) / Path(f"score_{score_val[0]:.4f}_{self.now()}.pt"), model=self.model, optim=self.optim)
                    if self.best.copy and 'score' in monitor:
                        self.best.score_model  = copy.deepcopy(self.model)

                for callback in self.callbacks:
                    callback.on_validation_epoch_end(self, self.model)

            if period_plot > 0 and (epoch % period_plot == 0 or epoch == epochs):
                if len(self.hist.trn.epochs) > 1:
                    self.plot()
                    for callback in self.callbacks:
                        callback.on_after_plot(self, self.model)
                self.stat()

            if self.folders.points and 'points' in monitor and (epoch % period_points == 0 or epoch == epochs):
                for callback in self.callbacks:
                    callback.on_save_checkpoint(self, self.model, {})
                score_val = score_val or [0]
                score_trn = score_trn or [0]
                self.save(Path(self.folders.points) / Path(f"points_{self.now()}_score_val_{score_val[0]:.4f}_trn_{score_trn[0]:.4f}_loss_val_{loss_val}_trn_{loss_trn}.pt"), model=self.model, optim=self.optim)

            self.step_schedulers(1, samples_trn)

            if samples is not None:
                samples -= samples_trn
                if samples <= 0:
                    self.plot()
                    self.stat()
                    break

            for callback in self.callbacks:
                callback.on_epoch_end(self, self.model)

            if patience is not None and patience > 0 and  epoch - last_best > patience:
                print(f"\n!!! Stop on patience={patience}. Epoch:{epoch}, last best score epoch:{self.hist.val.best.score_epochs}, best loss epoch:{self.hist.val.best.loss_epochs}")
                if period_plot > 0 and len(self.hist.trn.epochs) > 1:                
                    self.plot()
                    for callback in self.callbacks:
                        callback.on_after_plot(self, self.model)
                    self.stat()                
                break

        if period_plot <= 0:
            self.stat()

        for callback in self.callbacks:
            callback.on_fit_end(self, self.model)


    #---------------------------------------------------------------------------

    def plot(self, view=None, hist=None):
        """
        Plot training history
        """
        view = view or self.view
        hist = hist or self.hist
        plot_history(hist, view)

    #---------------------------------------------------------------------------

    def stat(self, newline=True):
        if newline:
            print()
        if self.hist.val.best.score is not None:
            print(f"validation score={self.hist.val.best.score:.6f}, loss={self.hist.val.best.loss:.6f};  epochs={self.hist.epochs}, samples={self.hist.samples}, steps={self.hist.steps}")
        elif self.hist.trn.best.score is not None:
            print(f"validation loss={self.hist.val.best.loss:.6f};  epochs={self.hist.epochs}, samples={self.hist.samples}, steps={self.hist.steps}")

        t_steps = f"{self.hist.time.trn*1_000/self.hist.steps:.2f}"   if self.hist.steps > 0 else "???"
        t_sampl = f"{self.hist.time.trn*1_000_000/self.hist.samples:.2f}" if self.hist.samples > 0 else "???"
        t_epoch = f"{self.hist.time.trn/self.hist.epochs:.2f}" if self.hist.epochs > 0 else "???"
        print(f"times=(trn:{self.hist.time.trn/60:.2f}, val:{self.hist.time.val/60:.2f})m,  {t_epoch} s/epoch, {t_steps} s/10^3 steps,  {t_sampl} s/10^6 samples")

    #---------------------------------------------------------------------------

    def best_score(self, best, score):
        return score is not None  and len(score) \
                and (best is None \
                or (best < score[0] and     self.score_max) \
                or (best > score[0] and not self.score_max) )

    #---------------------------------------------------------------------------

    def now(self):
        return datetime.datetime.now().strftime("%m.%d_%H-%M-%S")

    #---------------------------------------------------------------------------

    def add_hist(self, hist, batch_size, samples, steps, tm, loss, score, lr):
            hist.epochs    .append(self.hist.epochs)
            hist.samples   .append(self.hist.samples)
            hist.steps     .append(self.hist.steps)
            hist.samples_epoch.append(samples)
            hist.steps_epoch  .append(steps)
            hist.batch_size   .append(batch_size)
            hist.times        .append(tm)
            hist.lr           .append(lr)
            hist.losses         .append(loss)
            if score is not None and len(score):
                hist.scores.append(score[0].item())

    #---------------------------------------------------------------------------

    def save(self, fname, model = None, optim=None, info=""):
        try:
            model = model or self.model
            assert model is not None, "You don't have a model!"

            cfg = model.cfg if hasattr(model, "cfg") else Config()
            Path(fname).parent.mkdir(parents=True, exist_ok=True)            
            state = {
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
                    state['state_dict'] = optim.state_dict()
            torch.save(state, fname)
        except:
            print(f"Something  wrong in the function trainer.save fname: '{fname}'")

    #---------------------------------------------------------------------------

    def load(fname, ClassModel):
        """
        Static method of created trainer with loaded model.

        Example
        ------------ 
        trainer.save("cur_model.pt")
        ...
        new_trainer = Trainer.load("cur_model.pt")
        new_trainer.plot()
        """        
        try:
            state = torch.load(fname, map_location='cpu')
            print(f"info:  {state.get('info', '???')}")
            print(f"date:  {state.get('date', '???')}")
            print(f"model: {state.get('class','???')}")            
            print(f"optim: {state.get('optim','???')}")
            print(f"       {state.get('optim_cfg','???')}")
        except:
            print(f"Cannot open file: '{fname}'")
            return None
        
        
        assert type(state) == dict,  f"Apparently this model was not saved by the trainer: state:{type(state)}"
        assert 'model'  in state,    f"Apparently this model was not saved by the trainer: no 'model' in state: {list(state.keys())}"
        assert 'config' in state,    f"Apparently this model was not saved by the trainer: no 'config' in state: {list(state.keys())}"
            
        trainer = Trainer(None, None)
         
        trainer.model = ClassModel(state['config'])
        trainer.model.load_state_dict(state['model'])

        #trainer.optim.load_state_dict(state)

        trainer.hist(state['hist'])
        trainer.view(state['view'])

        return trainer
