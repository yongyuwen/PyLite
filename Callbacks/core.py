'''
TO-DO
1. Check implementation of Smoothed_LR_Find -> Adding debiasing
'''

from ..Utils import *


class Callback():
    _order=0
    def set_runner(self, run): self.run=run

    def __getattr__(self, k): return getattr(self.run, k)

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')


class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()


class TestCallback(Callback):
    _order=1
    def after_step(self):
        print(self.n_iter)
        if self.n_iter>=10: raise CancelTrainException()


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
        # self.smoothed_losses = []

    def after_batch(self):
        if not self.in_train: return
        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())
        # self.smoothed_losses.append(self.loss.detach().cpu()*alpha + (1-alpha) * self.losses[-2] if len(self.losses)> 1 else self.loss.detach().cpu())       

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])
        
    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs    = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])

    # def plot_smoothed(self, skip_last=0, pgid=-1):
    #     losses = [o.item() for o in self.smoothed_losses]
    #     lrs    = self.lrs[pgid]
    #     n = len(losses)-skip_last
    #     plt.xscale('log')
    #     plt.plot(lrs[:n], losses[:n])


class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9
        
    def begin_batch(self): 
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups: pg['lr'] = lr
            
    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            print(f"Exited here {self.n_iter}")
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss

class Smoothed_LR_Find(LR_Find):
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10, alpha=0.5):
        super().__init__(max_iter, min_lr, max_lr)
        self.smoothed_loss = None
        self.alpha = alpha

    def after_step(self):
        if self.smoothed_loss:
            self.smoothed_loss = self.loss*self.alpha + (1-self.alpha)*self.smoothed_loss
        else:
            self.smoothed_loss = self.loss

        if self.n_iter>=self.max_iter or self.smoothed_loss>self.best_loss*3:
            print(f"Exited here {self.n_iter}")
            raise CancelTrainException()
        if self.smoothed_loss < self.best_loss: self.best_loss = self.smoothed_loss