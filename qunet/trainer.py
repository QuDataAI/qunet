"""
Предполагается следующая абстракция:
Модель получает ненулевой кортеж  небходимых данных data=(X,Y,...):
В forward она возвращает кортеж  (float, тензор, тензор):
    Model.forward(data):    
        return output, loss, score
где
    loss   : float = ошибка по батчу с графом для обратного распространения (скаляр!)
    output : (B,*) результат работы модели (без графа), B - число примеров в батче
    score  : None or (B,1) or (B,n) - метрики качества работы (без графа)
Если метрик score нет, то  модель для них должна вернуть None. Если метрика одна - тензор (B,1).
Если нужны ошибки по каждому примеру, помещаем их в score, но loss всё равно оставляем скалярным.

Внимание: 
    Если выход модели один, после слоя Linear тензор имеет форму (B,1) 
    Поэтому таргетные данные также должны иметь форму (B,1), иначе получим неверный loss.
    X, = torch.arange(5).view(-1,1).to(torch.float32),  torch.arange(5).to(torch.float32)
    loss = (X-Y).pow(2).mean()  # 4 так как (B,1) - (B,) = (B,1) - (1,B) = (B,B)
"""
import os, math, copy, time, datetime
from   tqdm.auto import tqdm
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from .optim   import LineScheduler, ExpScheduler, CosScheduler
from .plotter import Plotter

class Trainer:
    """
    Универсальный класс обучения модели.
    Любой метод можно, естественно, переопределить наследованием или у экземпляра.
    """
    def __init__(self, model, data_trn, data_val, score_max=False) -> None:
        self.model = model
        
        self.optim = None
        self.scheduler = ExpScheduler()
        self.data_trn = data_trn
        self.data_val = data_val

        self.score_max = score_max      # метрика должна быть максимальной (например accuracy)

        self.best_loss_val  = None      # лучшее значение валидационной ошибки
        self.best_loss_trn  = None      # лучшее значение тренировочная ошибки
        self.best_score_val = None      # лучшее значение валидационной метрики (первой)
        self.best_score_trn = None      # лучшее значение тенировочной метрики (первой)
        self.samples_best_loss_val = 0  # когда была лучшая валидационная ошибка
        self.samples_best_loss_trn = 0  # когда была лучшая тренировочная ошибка
        self.samples_best_score_val= 0  # когда была лучшая валиадционная метрика        
        self.samples_best_score_trn= 0  # когда была лучшая тренировочная метрика

        self.samples  = 0               # число примеров в режиме обучения
        self.steps    = 0               # число шагов градиентного спуска

        # история обучения и валидации
        self.hist_trn = {'samples':[], 'steps':[], 'batch_size':[], 'lr':[], 'samples_epoch':[], 'steps_epoch':[], 'time':[], 'loss':[], 'score':[], 'best_loss':[], 'best_score':[] }
        self.hist_val = {'samples':[], 'steps':[], 'batch_size':[], 'lr':[], 'samples_epoch':[], 'steps_epoch':[], 'time':[], 'loss':[], 'score':[], 'best_loss':[], 'best_score':[] }
        self.labels   = []

        self.time_trn = 0               # общее время тренировки
        self.time_val = 0               # общее время валидациии

        self.model_best_score = None    # копия лучшей модели по score
        self.copy_best_model  = False   # копировать лучшую модель
        
        self.plotter = Plotter()        # класс рисования графиков

        self.cfg = {
            'folder_loss'  : None,      # папка для сохранения лучших val loss  моделей
            'folder_score' : None,      # папка для сохранения лучших val score моделей
            'folder_checks': None,      # папка для сохранения чекпоинтов
            'samples_unit_power':  6,   # единицы измерения числа примеров 10^samples_unit_power
            'x_min':         0,         # дипазон графика в samples по x (если < 0 последние x_min sampes)
            'x_max':         None,
            'plot_loss': {
                'show':  True,
                'y_min': None, 'y_max': None, 
                'ticks': None,
            },
            'plot_score': {
                'show':  True,
                'y_min': None, 'y_max': None, 
                'ticks': None,
            }
        }
        
    #---------------------------------------------------------------------------

    def set_optimizer(self, optim):
        """ Установить текущий оптимизатор """
        self.optim = optim
        self.scheduler.optim = optim

    #---------------------------------------------------------------------------

    def add_scheduler(self, scheduler):
        """ 
        Установить текущий оптимизатор. todo: список шедулеров
        """
        assert self.optim is not None, "Define the optimizer first"
        self.scheduler = scheduler
        self.scheduler.optim = self.optim

    #---------------------------------------------------------------------------

    def fit(self, epoch, model, data,  train=True, accumulate=1, verbose=1):
        """
        Args:
            * train      - True: режим тренировки, иначе валидации
            * accumulate - аккумулировать градиент для стольки батчей перед сдвигом оптимизатора;
                           используем, когда большой батч или граф не помещаются в GPU        
        """
        self.model.train(train)              # режим обучение или тестирование
        torch.set_grad_enabled(train)        # строить или нет вычислительный граф

        if train:
            self.optim.zero_grad()            # обнуляем градиенты

        samples, steps, beg, lst = 0, 0, time.time(), time.time()
        counts_all, losses_all,  scores_all = torch.empty((0,1)), None,  None
        for b, batch in enumerate(data):
            loss, _, scores = model(batch)
                        
            if train:
                loss.backward()               # вычисляем градиенты
                if (b+1) % accumulate == 0:
                    self.optim.step()         # подправляем параметры
                    self.optim.zero_grad()    # обнуляем градиенты
                    steps      += 1           # число сделанных шагов
                    self.steps += 1           # число сделанных шагов за всё время
                self.samples += len(batch[0]) # число просмотренных примеров за всё время

            samples += len(batch[0])          # число просмотренных примеров за эпоху
            losses_all = loss.detach() if losses_all is None else torch.vstack([losses_all, loss.detach()])
            if scores is not None:
                scores = scores.detach().mean(dim=0)
                scores_all = scores if scores_all is None else torch.vstack([scores_all, scores])
            counts_all = torch.vstack([counts_all, torch.Tensor([len(batch[0])])])

            if verbose and (time.time()-lst > 1 or b+1 == len(data) ):
                lst = time.time()
                self.fit_progress(epoch, train, (b+1)/len(data),
                                  losses_all, scores_all, counts_all, samples, steps, time.time()-beg)

        if train: self.time_trn += (time.time()-beg)
        else:     self.time_val += (time.time()-beg)

        if scores_all is not None:
            scores_all = scores_all.cpu()
        return losses_all.cpu(), scores_all, counts_all, (samples,steps,time.time()-beg)
    
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
        steps, samples = max(steps, 1), max(samples, 1)             # just in case
        st = ""
        if score is not None and len(score):
            st += f"score={score[0]:.4f} "                          # главная метрика
            if len(score) > 1: st += "("                            # вспомогательные
            for i in range(1, len(score)):
                st += f"{score[i]:.4f}" + (", " if i+1 < len(score) else ") ")
        st += f"loss={loss:.4f} "                
        print(f"\r{epoch:3d}{'t' if train else 'v'}[{100*done:3.0f}%]  {st}  samples={samples} steps={steps}  time={1e3*tm/steps:.3}ms/step  {1e6*tm/samples:.2f}s/1e6", end="  ")

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

        for b, batch in enumerate(data):
            samples += len(batch[0])                  # число просмотренных примеров
            loss, output, scores = model(batch)
            
            losses_all = loss.detach() if losses_all is None else torch.vstack([losses_all, loss.detach()])
            if scores is not None:
                scores = scores.detach().mean(0)
                scores_all = scores if scores_all is None else torch.vstack([scores_all, scores])
            counts_all = torch.vstack([counts_all, torch.Tensor([len(batch[0])])])
            output_all = output if output_all is None else torch.vstack([output_all, output.detach()])

            if verbose and (time.time()-lst > 1 or b+1 == len(data) ):
                lst = time.time()                                
                self.fit_progress(0, False, (b+1)/len(data), losses_all, scores_all, counts_all, samples, steps, time.time()-beg)

        if scores is not None:
            scores_all = scores_all.cpu()
        return output_all.cpu(),  losses_all.cpu(), scores_all

    #---------------------------------------------------------------------------

    def run(self, 
            epochs = 100,             # число эпох для обучения (проходов одного пака data_trn)
            pre_val=False,            # перед началом обучения сделать валидацию            
            period_plot=100,          # через сколько эпох выводить график обучения 
            period_checks=1,          # через сколько эпох делать чекпоинты (сохранять модель)
            period_val=1,             # через сколько эпох делать валидацию
            period_val_beg = 4,       # период валидации на первых samples_beg примерах
            samples_beg = None,       # потом включается period_val
            stop_after_samples=None): # остановится после этого числа примеров
        """
        Args:
            * epochs               - число эпох для обучения (полных проходов data_trn)
            * pre_val              - перед началом обучения сделать валидацию
            * period_val           - через сколько эпох делать валидацию
            * period_plot          - через сколько эпох выводить график обучения
            * period_checks        - через сколько эпох делать чекпоинты (сохранять модель)
            * period_val_beg = 4,  - период валидации на первых samples_beg примерах
            * samples_beg = None,  - потом включается period_val                        
            * stop_after_samples   - остановится после этого числа примеров
        """
        assert self.optim is not None, "Define the optimizer first"
        self.scheduler.optim = self.optim

        if pre_val:
            losses, scores, counts, (samples_val, steps_val, tm_val) = self.fit(0, self.model, self.data_val, train=False)
            loss_val, score_val = self.mean(losses, scores, counts)
            self.add_hist(self.hist_val, self.data_val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, self.scheduler.get_lr())
            print()
        
        #for epoch in tqdm(range(1, epochs+1)):
        for epoch in range(1, epochs+1):
            losses, scores, counts, (samples_trn,steps_trn,tm_trn) = self.fit(epoch, self.model, self.data_trn, train=True)
            loss_trn, score_trn = self.mean(losses, scores, counts)
            lr = self.scheduler.get_lr()
            self.add_hist(self.hist_trn, self.data_trn.batch_size, samples_trn, steps_trn, tm_trn, loss_trn, score_trn, lr)

            if self.best_loss_trn is None or self.best_loss_trn > loss_trn:
                self.best_loss_trn = loss_trn
                self.samples_best_loss_trn = self.samples
                self.hist_trn['best_loss'].append( (self.samples, self.steps, loss_trn) )
            if self.best_score_trn is None or self.best_score_trn > score_trn[0]:
                self.best_score_trn = score_trn[0]
                self.samples_best_score_trn = self.samples
                self.hist_trn['best_score'].append((self.samples, self.steps, score_trn[0].item()))

            period = period_val_beg if samples_beg and  self.samples < samples_beg else period_val
            if  epoch % period == 0 or epoch == epochs:                
                losses, scores, counts, (samples_val,steps_val,tm_val) = self.fit(epoch, self.model, self.data_val, train=False)
                loss_val, score_val = self.mean(losses, scores, counts)
                self.add_hist(self.hist_val, self.data_val.batch_size, samples_val, steps_val, tm_val, loss_val, score_val, lr)

                # save best validation loss:
                if self.best_loss_val is None or self.best_loss_val > loss_val:       
                    self.best_loss_val = loss_val
                    self.samples_best_loss_val = self.samples
                    self.hist_val['best_loss'].append((self.samples, self.steps, loss_val))
                    if self.cfg.get('folder_loss', False):
                        self.save(self.model, folder=self.cfg['folder_loss'], prefix=f"loss_{loss_val:.4f}_{self.now()}")

                # save best validation score[0]
                if score_val is not None  and len(score_val) \
                and (self.best_score_val is None \
                     or (self.best_score_val < score_val[0] and self.score_max) \
                     or (self.best_score_val > score_val[0] and not self.score_max)                     
                ):  
                    self.best_score_val = score_val[0]
                    self.samples_best_score_val = self.samples
                    self.hist_val['best_score'].append( (self.samples, self.steps, score_val[0].item()) )
                    if self.cfg.get('folder_score', False):
                        self.save(self.model, folder=self.cfg['folder_score'], prefix=f"score_{score_val[0]:.4f}_{self.now()}")

                    if self.copy_best_model:         # копия лучшей модели по score
                        self.model_best_score = copy.deepcopy(self.model) 

            if epoch % period_plot == 0 or epoch == epochs:
                print(f"\ntime = (trn:{self.time_trn/60:.2f}, val:{self.time_val/60:.2f}, tot:{(self.time_trn+self.time_val)/60:.2f})m  lr:{self.scheduler.get_lr():.1e}")
                self.plot()

            if self.cfg.get('folder_checks', False) and (epoch % period_checks == 0 or epoch == epochs):
                self.save(self.model, folder="", prefix="check_"+self.now())

            if stop_after_samples is not None:
                stop_after_samples -= samples_trn
                if stop_after_samples <= 0:
                    self.plot()
                    break

            self.scheduler.step(samples_trn)

    def plot(self):
        self.plotter.plot(self.cfg, self.model, data=dict(
                            hist_val=self.hist_val, hist_trn=self.hist_trn, labels=self.labels,
                            samples = self.samples, steps = self.steps)
                        )

    #---------------------------------------------------------------------------

    def now(self):
        return datetime.datetime.now().strftime("%m-%d %H:%M:%S")

    #---------------------------------------------------------------------------

    def add_hist(self, hist, batch_size, samples, steps, tm, loss, score, lr):
            hist['samples']   .append(self.samples)
            hist['steps']     .append(self.steps)            
            hist['samples_epoch']   .append(samples)
            hist['steps_epoch']     .append(steps)
            hist['batch_size'].append(batch_size)
            hist['time']      .append(tm)
            hist['lr']        .append(lr)
            hist['loss']      .append(loss)
            if score is not None and len(score):
                hist['score'] .append(score[0].item())

    #---------------------------------------------------------------------------

    def set_scheduler(self, enable=True, kind='exp', lr0=1e-3 ):
        pass

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

            'data': {                                     # история обучения
                'steps':           self.steps,            
                'samples':         self.samples,
                'hist_trn':        self.hist_trn,
                'hist_val':        self.hist_val,
                'labels':          self.labels,             
                'best_loss_val':   self.best_loss_val,
                'best_loss_trn':   self.best_loss_trn,                          
                'best_score_val':  self.best_score_val,
                'best_score_trn':  self.best_score_trn,
            }
        }    
        torch.save(state, fname)
        
