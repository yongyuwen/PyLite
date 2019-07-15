# TO-DO
'''
1. Fix passing of fname to download data
'''

from .imports import *

PathOrStr = Union[Path,str]

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
    print(url)
    s = requests.Session()
    s.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try: file_size = int(u.headers["Content-Length"])
    except: print("File size info not found")

    with open(dest, 'wb') as f:
        print(f"Writing to {dest}")
        nbytes = 0
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            timeout_txt =(f'\n Download of {url} has failed after {retries} retries\n'
                          f' Fix the download manually:\n'
                          f'And re-run your code once the download is successful\n')
            print(timeout_txt)
            import sys;sys.exit(1)

