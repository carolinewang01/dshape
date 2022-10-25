import numpy as np 
import argparse
from contextlib import contextmanager
import sys, os
from utils.load_confs import load_parameters, load_paths

params = load_parameters()
paths = load_paths()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

