from .utils import *
from .data import *

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

def get_mnist_data():
    path = download_data(MNIST_URL, 'data/mnist', ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def get_ds(x_train, y_train, x_valid, y_valid):
    return Dataset(x_train, y_train), Dataset(x_valid, y_valid)

def get_dls(train_ds, valid_ds, bs, **kwargs):
    '''
    Allows you to pass specific kwargs using 'train_' and 'valid_'
    to specify which kwargs to be passed into train or valid. If no such
    prefix is defined, the kwarg will be passed into both DataLoaders
    '''
    train_kwargs = {k.replace('train_', ''): kwargs[k] for k in kwargs if 'valid_' not in k}
    valid_kwargs = {k.replace('valid_', ''): kwargs[k] for k in kwargs if 'train_' not in k}
    shuffle = False if 'sampler' in train_kwargs else True
    return (DataLoader(train_ds, batch_size=bs, shuffle=shuffle, **train_kwargs), 
            DataLoader(valid_ds, batch_size=bs*2, **valid_kwargs))

def get_model(data, lr=0.5, nh=50):
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,data.c))
    return model, optim.SGD(model.parameters(), lr=lr)

def get_model_func(lr=0.5): return partial(get_model, lr=lr)

def create_learner(model_func, loss_func, data, cbs=None, cb_funcs=None):
    return Learner(*model_func(data), loss_func, data, cbs, cb_funcs)

