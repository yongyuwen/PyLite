# TO-DO
'''
1. Fix passing of fname to download data.
2. Progress bar for downloading files
'''

from .imports import *

PathOrStr = Union[Path,str]

# Stats
class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
    
    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

# Formaters
_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, Iterable): return list(o)
    return [o]

# Data Downloaders
MB_DENOM = 1048576 

def download_data(url:str, fname:PathOrStr, data:bool=True, ext:str='.tgz') -> Path:
    "Download `url` to destination `fname`."
    fname = Path(fname)
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}{ext}', fname)
    return fname

def download_url(url:str, dest:str, overwrite:bool=False, chunk_size=1024*1024, 
                 timeout=4, retries=5)->None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return
    s = requests.Session()
    s.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    file_size = None
    try: file_size = int(u.headers["Content-Length"])
    except: print("File size info not found")

    with open(dest, 'wb') as f:
        print(f"Writing to {dest}")
        nbytes = 0
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                f.write(chunk)
                print(f"Downloaded {nbytes/MB_DENOM:.2f}/{file_size/MB_DENOM:.2f}MB ({nbytes/file_size*100:.2f}%)", end="\r")
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            timeout_txt =(f'\n Download of {url} has failed after {retries} retries\n'
                          f' Fix the download manually:\n'
                          f'And re-run your code once the download is successful\n')
            print(timeout_txt)
            import sys;sys.exit(1)

