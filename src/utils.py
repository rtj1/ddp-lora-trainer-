import random, time, yaml
import numpy as np
import torch
from dataclasses import dataclass
from torch.distributed import is_initialized, get_world_size, get_rank

@dataclass
class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        return AttrDict(yaml.safe_load(f))

def override_from_kv(cfg: AttrDict, overrides):
    for kv in overrides:
        if "=" not in kv: 
            continue
        key, val = kv.split("=", 1)
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node: node[k] = {}
            node = node[k]
        v = val
        if val.lower() in ("true","false"): v = val.lower()=="true"
        else:
            try:
                v = float(val) if "." in val else int(val)
            except ValueError:
                v = val
        node[keys[-1]] = v
    return cfg

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ddp_info():
    if is_initialized():
        return get_rank(), get_world_size()
    return 0,1

def is_main_process():
    r,_ = ddp_info()
    return r==0

def barrier():
    if is_initialized():
        torch.distributed.barrier()

class ThroughputMeter:
    def __init__(self, window=50):
        self.window = window; self.hist=[]
    def update(self, tokens:int, dt:float):
        if dt>0:
            self.hist.append(tokens/dt)
            if len(self.hist)>self.window: self.hist.pop(0)
    def avg_toks_per_sec(self):
        return sum(self.hist)/len(self.hist) if self.hist else 0.0
