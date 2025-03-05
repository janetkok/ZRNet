""" Model / state_dict utils

- Hacked together by / Copyright 2020 Ross Wightman
- Med3D: Transfer Learning for 3D Medical Image Analysis(Chen, Sihong and Ma, Kai and Zheng, Yefeng) 2019
- modified by Janet Kok for multi channel 3D images on classification task
"""
import torch
from torch import nn
import numpy as np

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()



