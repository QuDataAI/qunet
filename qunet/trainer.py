import os, math, copy, time, datetime
from   tqdm.auto import tqdm
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from .optim   import Scheduler
from .plots   import plot_history

class Trainer:
    """
    Generic model training class.
    Any method can, of course, be overridden by inheritance or by an instance.
    """
    def __init__(self, model, data_trn, data_val, score_max=False) -> None:
        """
        Args:
            * model     - model for traininig;
            * data_trn  - training data (Data or DataLoader instance);
            * data_val  - data for validation (instance of Data or DataLoader); may be missing;
            * score_max - consider that the metric (the first column of the second tensor returned by the function `metrics` of the model ); should strive to become the maximum (for example, so for accuracy).            
        """
        self.model      = model
        self.data_trn   = data_trn
        self.data_val   = data_val
        self.score_max = score_max       # метрика должна быть максимальной (например accuracy)

        self.device     = "cuda" if torch.cuda.is_available() else "cpu"         
        self.dtype      = torch.float32 # see training large models;
        self.optim      = None
        self.schedulers = []             # список шедулеров для управления обучением
        self.scheduler  = Scheduler()    # текущий шедулер

        self.copy_best_score_model = False  # to copy the best model by val score
        self.copy_best_loss_model  = False  # to copy the best model by val loss

        self.best_score_model = None     # copy of the best model by val score
        self.best_loss_model  = None     # copy of the best model by val loss
         
        self.folder_loss   = None        # folder to save the best val loss models
        self.folder_score  = None        # folder to save the best val score models
        self.folder_checks = None        # folder to save checkpoints        

        # -------------------------------- настройки для построения графиков ошибок и метрик
        self.view = {                    
            'w'            : 12,         # plt-plot width
            'h'            :  5,         # plt-plot height
            'count_units'  : 1e6,        # units for number of samples
            'time_units'   : 's',        # time units: ms, s, m, h

            'x_min'        : 0,          # minimum value in samples on the x-axis (if < 0 last x_min samples)
            'x_max'        : None,       # maximum value in samples on the x-axis (if None - last)

            'loss': {                                
                'show'  : True,          # show loss subplot
                'y_min' : None,          # fixing the minimum value on the y-axis
                'y_max' : None,          # fixing the maximum value on the y-axis
                'ticks' : None,          # how many labels on the y-axis
                'lr'    : True,          # show learning rate
                'labels': True,          # show labels (training events)                
                'trn_checks': True,          # show the achievement of the minimum training loss (dots)
                'val_checks': True,          # show the achievement of the minimum validation loss (dots)

            },
            'score': {                    
                'show'  : True,          # show score subplot    
                'y_min' : None,          # fixing the minimum value on the y-axis
                'y_max' : None,          # fixing the maximum value on the y-axis
                'ticks' : None,          # how many labels on the y-axis
                'lr'    : True,          # show learning rate                
                'labels': True,          # show labels (training events)
                'trn_checks': True,      # show the achievement of the optimum training score (dots)                
                'val_checks': True,      # show the achievement of the optimum validation score (dots)                
            }
        }

        # -------------------------------- история и текущие метрики процесса обучения модели
        self.hist = {                    # история обучения и валидации:
            'samples': 0,                # число примеров в режиме обучения
            'steps'  : 0,                # число шагов градиентного спуска

            'time_trn': 0,               # общее время тренировки
            'time_val': 0,               # общее время валидациии

            'best':  {
                'loss_val': None,        # лучшее значение валидационной ошибки
                'loss_trn': None,        # лучшее значение тренировочная ошибки
                'score_val': None,       # лучшее значение валидационной метрики (первой)
                'score_trn': None,       # лучшее значение тенировочной метрики (первой)

                'samples_loss_val': 0,   # когда была лучшая валидационная ошибка
                'samples_loss_trn': 0,   # когда была лучшая тренировочная ошибка
                'samples_score_val': 0,  # когда была лучшая валиадционная метрика
                'samples_score_trn': 0,  # когда была лучшая тренировочная метрика
            },

            'trn'    : {'samples':[], 'steps':[], 'batch_size':[], 'lr':[], 'samples_epoch':[], 'steps_epoch':[], 'time':[], 'loss':[], 'score':[], 'best_loss':[], 'best_score':[] },
            'val'    : {'samples':[], 'steps':[], 'batch_size':[], 'lr':[], 'samples_epoch':[], 'steps_epoch':[], 'time':[], 'loss':[], 'score':[], 'best_loss':[], 'best_score':[] },
            'labels' : []
        }    
        self.hist['params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)        

    #---------------------------------------------------------------------------

    def add_label(self, text):
        """ Добавить пометку для графиков обучения """
        h = self.hist
        self.hist['labels'].append( [ text, h['samples'], h['steps'], h['time_trn'], h['time_val'] ] )

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

    def step_schedulers(self, samples_trn):
        for sch in self.schedulers:
            if sch.enable:
                sch.step(samples_trn)
                self.scheduler = sch
                break

    #---------------------------------------------------------------------------

    def to_device(self, X):
        if type(X) is list or type(X) is tuple:
            for i in range(len(X)):
                X[i] = X[i].to(self.device)
        else:
            X = X.to(self.device)
        return X

    #---------------------------------------------------------------------------

    def fit(self, epoch, model, data,  train=True, accumulate=1, verbose=1):
        """
        Args:
            * train      - True: режим тренировки, иначе валидации
            * accumulate - аккумулировать градиент для стольки батчей перед сдвигом оптимизатора;
                           используем, когда большой батч или граф не помещаются в GPU
        https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
        """
        self.model.train(train)                     # режим обучение или тестирование
        torch.set_grad_enabled(train)               # строить или нет вычислительный граф

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        if train:
            self.optim.zero_grad()                  # обнуляем градиенты

        samples, steps, beg, lst = 0, 0, time.time(), time.time()
        counts_all, losses_all,  scores_all = torch.empty(0,1), None,  None
        for b, (x, y_true) in enumerate(data):
            num = len(x[0]) if type(x) is list or type(x) is tuple else len(x)
            x, y_true = self.to_device(x), self.to_device(y_true)

            if scaler is None:
                y_pred = model(x)
                loss, scores = model.metrics(x, y_pred, y_true)
            else:
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    y_pred = model(x)
                    loss, scores = model.metrics(x, y_pred, y_true)

            if train:
                if scaler is None:
                    loss.backward()   # вычисляем градиенты
                else:
                    scaler.scale(loss).backward()   # вычисляем градиенты
                if (b+1) % accumulate == 0:
                    if scaler is None:
                        self.optim.step()
                    else:
                        scaler.step(self.optimizer) # подправляем параметры
                        scaler.update()             # Updates the scale for next iteration
                    self.optim.zero_grad()          # обнуляем градиенты
                    steps      += 1                 # шагов за эпоху
                    self.hist['steps'] += 1         # шагов за всё время
                self.hist['samples'] += num         #  примеров за всё время

            samples += num                          # просмотренных примеров за эпоху
            losses_all = loss.detach() if losses_all is None else torch.vstack([losses_all, loss.detach()])
            if scores is not None:
                scores = scores.detach().mean(dim=0)
                scores_all = scores if scores_all is None else torch.vstack([scores_all, scores])
            counts_all = torch.vstack([counts_all, torch.Tensor([num])])

            if verbose and (time.time()-lst > 1 or b+1 == len(data) ):
                lst = time.time()
                self.fit_progress(epoch, train, (b+1)/len(data),
                                  losses_all, scores_all, counts_all, samples, steps, time.time()-beg)

        if train: self.hist['time_trn'] += (time.time()-beg)
        else:     self.hist['time_val'] += (time.time()-beg)

        if scores_all is not None:
            scores_all = scores_all.cpu()
        return losses_all.cpu(), scores_all, counts_all, (samples, steps, time.time()-beg)

    #---------------------------------------------------------------------------

    def mean(self, losses, scores, counts):
        """ Вычислить среднее по всей эпохе """
        loss  = ((losses.detach().cpu() * counts).sum(dim=0) / counts.sum()).item()
        if scores is not None:
            scores = ((scores.detach().cpu() * counts).sum(dim=0) / counts.sum())
        return (loss, scores)

    #---------------------------------------------------------------------------

    def fit_progress(self, epoch, train, done, losses, scores, counts, samples, steps, tm):
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
        st += f"loss={loss:.4f} "

        t_unit, t_unit_scale,  c_unit, c_unit_power = self.unit_scales()
        print(f"\r{epoch:3d}{'t' if train else 'v'}[{100*done:3.0f}%]  {st}  samples={samples} steps={steps}  time={(0.0 if steps==0 else 1e3*tm/(t_unit_scale*steps)):.3}{t_unit}/step  {c_unit*tm/(t_unit_scale*samples):.2f}{t_unit}/10^{c_unit_power:.0f}", end="                ")

    #---------------------------------------------------------------------------

    def unit_scales(self):
        """ Единичы измерения числа примеров и времени """
        view = self.view
        t_unit = view['time_units'] if 'time_units' in view and view['time_units'] in ['ms','s','m','h']  else 's'
        t_unit_scale = dict(ms=1e-3, s=1, m=60, h=3600)[t_unit]
        c_unit = view['count_units'] if view.get('count_units',0) > 0  else 1
        c_unit_power = round(np.log10(c_unit), 0)
        return t_unit, t_unit_scale,  c_unit, c_unit_power

    #---------------------------------------------------------------------------

    def predict(self, model, data, verbose:bool = True, whole=False):
        """
        Вычислить предлсказание, ошибку и метрику для каждого примера в data
        todo: выдели сразу память
        """
        self.model.train(False)          # режим тестирование
        torch.set_grad_enabled(False)    # вычислительный граф не строим
        data.whole = whole               # обычно по всем примерам (и по дробному батчу)

        samples, steps, beg, lst = 0, 0, time.time(), time.time()
        counts_all, losses_all, scores_all, output_all = torch.empty((0,1)), None,  None, None

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        for b, (x,y_true) in enumerate(data):
            if scaler is None:
                y_pred = model(x)
                loss, scores = model.metrics(x, y_pred, y_true)
            else:
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    y_pred = model(x)
                    loss, scores = model.metrics(x, y_pred, y_true)

            num = len(x[0]) if type(x) is list or type(x) is tuple else len(x)
            samples += num                      # число просмотренных примеров за эпоху
            losses_all = loss.detach() if losses_all is None else torch.vstack([losses_all, loss.detach()])
            if scores is not None:
                scores = scores.detach().mean(dim=0)
                scores_all = scores if scores_all is None else torch.vstack([scores_all, scores])
            counts_all = torch.vstack([counts_all, torch.Tensor([num])])

            if verbose and (time.time()-lst > 1 or b+1 == len(data) ):
                lst = time.time()
                self.fit_progress(0, False, (b+1)/len(data), losses_all, scores_all, counts_all, samples, steps, time.time()-beg)

        if scores is not None:
            scores_all = scores_all.cpu()
        return output_all.cpu(),  losses_all.cpu(), scores_all

    #---------------------------------------------------------------------------

    def run(self,  epochs =None,  samples=None,            
            pre_val:bool=False, period_val:int=1, period_plot:int=100,         
            period_checks:int=1, period_val_beg:int = 4, samples_beg:int = None,
            period_call:int = 0, callback = None): 
        """
        Args:
            * epochs               - number of epochs for training (passes of one data_trn pack). If not defined (None) works "infinitely".
            * samples              - if defined, then will stop after this number of samples, even if epochs has not ended
            * pre_val              - validate before starting training
            * period_val           - period after which validation run (in epochs)
            * period_plot          - period after which the training plot is displayed (in epochs)
            * period_call          - callback custom function call period
            * callback             - custom function called with period_info
            * period_checks        - period after which checkpoints are made and the current model is saved (in epochs)
            * period_val_beg = 4   - validation period on the first samples_beg examples
            * samples_beg = None   - the number of samples from the start, after which the validation period will be equal to period_val.
        """
        assert self.optim is not None, "Define the optimizer first"
        self.set_optim_schedulers()        
        self.model.to(self.device)
        if pre_val:
            losses, scores, counts, (samples_val, steps_val, tm_val) = self.fit(0, self.model, self.data_val, train=False)
            loss_val, score_val = self.mean(losses, scores, counts)
            self.add_hist(self.hist['val'], self.data_val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, self.scheduler.get_lr())
            print()

        epochs = epochs or 1_000_000
        #for epoch in tqdm(range(1, epochs+1)):
        for epoch in range(1, epochs+1):
            losses, scores, counts, (samples_trn,steps_trn,tm_trn) = self.fit(epoch, self.model, self.data_trn, train=True)
            loss_trn, score_trn = self.mean(losses, scores, counts)
            lr = self.scheduler.get_lr()
            self.add_hist(self.hist['trn'], self.data_trn.batch_size, samples_trn, steps_trn, tm_trn, loss_trn, score_trn, lr)

            if self.hist['best']['loss_trn'] is None or self.hist['best']['loss_trn'] > loss_trn:
                self.hist['best']['loss_trn'] = loss_trn
                self.hist['best']['samples_loss_trn'] = self.hist['samples']
                self.hist['trn']['best_loss'].append( (self.hist['samples'], self.hist['steps'], loss_trn) )

            if self.best_score(self.hist['best']['score_trn'], score_trn):            
                self.hist['best']['score_trn'] = score_trn[0]
                self.hist['best']['samples_score_trn'] = self.hist['samples']
                self.hist['trn']['best_score'].append((self.hist['samples'], self.hist['steps'], score_trn[0].item()))

            period = period_val_beg if samples_beg and  self.hist['samples'] < samples_beg else period_val
            if  (period and epoch % period == 0) or epoch == epochs:
                losses, scores, counts, (samples_val,steps_val,tm_val) = self.fit(epoch, self.model, self.data_val, train=False)
                loss_val, score_val = self.mean(losses, scores, counts)
                self.add_hist(self.hist['val'], self.data_val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, lr)

                # save best validation loss:
                if self.hist['best']['loss_val'] is None or self.hist['best']['loss_val'] > loss_val:
                    self.hist['best']['loss_val'] = loss_val
                    self.hist['best']['samples_loss_val'] = self.hist['samples']
                    self.hist['val']['best_loss'].append((self.hist['samples'], self.hist['steps'], loss_val))
                    if self.folder_loss:
                        self.save(self.model, folder=self.folder_loss, prefix=f"loss_{loss_val:.4f}_{self.now()}")
                    if self.copy_best_loss_model:
                        self.best_loss_model  = copy.deepcopy(self.model)
                
                if self.best_score(self.hist['best']['score_val'], score_val):
                    self.hist['best']['score_val'] = score_val[0]
                    self.hist['best']['samples_score_val'] = self.hist['samples']
                    self.hist['val']['best_score'].append( (self.hist['samples'], self.hist['steps'], score_val[0].item()) )
                    if self.folder_score:
                        self.save(self.model, folder=self.folder_score, prefix=f"score_{score_val[0]:.4f}_{self.now()}")
                    if self.copy_best_score_model:
                        self.best_score_model  = copy.deepcopy(self.model)

            if (period_plot and epoch % period_plot == 0) or epoch == epochs:
                self.run_progress()        
                plot_history(self.hist, self.view) 
            
            if callback and period_call and epoch % period_call == 0:                
                callback()

            if self.folder_checks and (epoch % period_checks == 0 or epoch == epochs):
                self.save(self.model, folder=self.folder_checks, prefix="check_"+self.now())

            self.step_schedulers(samples_trn)

            if samples is not None:
                samples -= samples_trn
                if samples <= 0:
                    self.run_progress()        
                    plot_history(self.hist, self.view) 
                    if callback:
                        callback()
                    break

    #---------------------------------------------------------------------------

    def run_progress(self):
        print(f"\ntime = (trn:{self.hist['time_trn']/60:.2f}, val:{self.hist['time_val']/60:.2f}, tot:{(self.hist['time_trn']+self.hist['time_val'])/60:.2f})m  lr:{self.scheduler.get_lr():.1e}")
    #---------------------------------------------------------------------------
            
    def best_score(self, best, score):
        return score is not None  and len(score) \
                and (best is None \
                or (best < score[0] and     self.score_max) \
                or (best > score[0] and not self.score_max) )

    #---------------------------------------------------------------------------

    def now(self):
        return datetime.datetime.now().strftime("%m-%d %H:%M:%S")

    #---------------------------------------------------------------------------

    def add_hist(self, hist, batch_size, samples, steps, tm, loss, score, lr):
            hist['samples']   .append(self.hist['samples'])
            hist['steps']     .append(self.hist['steps'])
            hist['samples_epoch']   .append(samples)
            hist['steps_epoch']     .append(steps)
            hist['batch_size'].append(batch_size)
            hist['time']      .append(tm)
            hist['lr']        .append(lr)
            hist['loss']      .append(loss)
            if score is not None and len(score):
                hist['score'] .append(score[0].item())

    #---------------------------------------------------------------------------

    def save(self, fname, model = None, optim=None, info=""):
        model = model or self.model
        cfg = model.cfg
        state = {
            'info':            info,
            'date':            datetime.datetime.now(),   # дата и время
            'config':          cfg,                       # конфигурация модели
            'model' :          model.state_dict(),        # параметры модели
            'optimizer':       optim.state_dict() if optim is not None else None,
            'hist':            self.hist,
        }
        torch.save(state, fname)

