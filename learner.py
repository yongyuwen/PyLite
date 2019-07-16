from .utils import *
from .Callbacks import *


class Learner():
    def __init__(
            self, 
            model, 
            opt, 
            loss_func, 
            data, 
            cbs=None, 
            cb_funcs=None):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
        
        self.cbs = [TrainEvalCallback()]
        self.add_callbacks(cbs, cb_funcs)
        self.stop = False
    
    def add_callbacks(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.cbs.extend([cb for cb in cbs if cb.name not in [cb.name for cb in self.cbs]])

    def one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            if self('begin_batch'): return
            self.pred = self.model(self.xb)
            if self('after_pred'): return
            self.loss = self.loss_func(self.pred, self.yb)
            if self('after_loss') or not self.in_train: return
            self.loss.backward()
            if self('after_backward'): return
            self.opt.step()
            if self('after_step'): return
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')


    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,yb in dl: self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs):
        self.epochs, self.loss = epochs, tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            if self('begin_fit'): return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad(): 
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                self('after_epoch')
        
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res