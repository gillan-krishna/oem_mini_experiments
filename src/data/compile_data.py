import zipfile
from pathlib import Path

def compile_data(DATA_ZIP:Path, OUTPATH:Path):
    with zipfile.ZipFile(DATA_ZIP, 'r') as fzip:
        fzip.extractall(path=OUTPATH)
    

if __name__ == '__main__':
    DATA_ZIP = Path('/home/ubuntu/hrl/oem_mini_experiments/data/raw/OpenEathMap_Mini.zip')
    OUTPATH = Path('data/processing')
    compile_data(DATA_ZIP=DATA_ZIP, OUTPATH=OUTPATH)