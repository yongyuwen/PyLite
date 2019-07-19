from .core import *

LOCAL_PATH = Path.cwd()

# URLS
MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
IMAGENETTE_160 = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz'

# Data Downloaders
MB_DENOM = 1048576

def untar_data(url, fname=None):
    fname = download_data(url)
    dest = Path(''.join(str(fname).split('.')[:-1]))
    if not fname.exists():
        tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest

def download_data(url:str, fname:PathOrStr=None, data:bool=True) -> Path:
    "Download `url` to destination `dest`."
    fname = Path(ifnone(fname, LOCAL_PATH/'data'/url.split('/')[-1]))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}', fname)
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
    except: 
        print("File size info not found")

    with open(dest, 'wb') as f:
        print(f"Writing to {dest}")
        nbytes = 0
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                f.write(chunk)
                if file_size:
                    print(f"Downloaded {nbytes/MB_DENOM:.2f}/{file_size/MB_DENOM:.2f}MB ({nbytes/file_size*100:.2f}%)", end="\r")
                else:
                    print(f"Downloaded {nbytes/MB_DENOM:.2f}", end="\r")
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            timeout_txt =(f'\n Download of {url} has failed after {retries} retries\n'
                          f' Fix the download manually:\n'
                          f'And re-run your code once the download is successful\n')
            print(timeout_txt)
            import sys;sys.exit(1)