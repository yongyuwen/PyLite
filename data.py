from .imports import *

from torch.utils.data import DataLoader

class Dataset():
    def __init__(self, x:torch.Tensor, y:torch.Tensor): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c
        
    @property
    def train_ds(self): return self.train_dl.dataset
        
    @property
    def valid_ds(self): return self.valid_dl.dataset

class TabularDataBunch(DataBunch):
    '''
    Automatically infers the number of classes from 
    the labels of the training set
    '''
    def __init__(self, train_dl, valid_dl, c=None):
        super().__init__(train_dl, valid_dl, c)
        if not self.c: self.c = self.train_ds.y.max().item() + 1