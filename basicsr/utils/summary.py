""" Summary utilities
"""
import csv
import os
from collections import OrderedDict



def update_summary(epoch, train_metrics, val_metrics=None, filename='out.csv', write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if val_metrics is not None:
        rowd.update([('val_' + k, v) for k, v in val_metrics.items()])


    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
