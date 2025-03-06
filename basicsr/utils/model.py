""" Model / state_dict utils
"""
import torch
from torch import nn
import numpy as np

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()



