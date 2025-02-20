#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math

import torch
from torch import nn


def FedAvg(w,num_data):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += (w[i][k]*num_data[i]/num_data[0])
        w_avg[k] = torch.div(w_avg[k], sum(num_data)/num_data[0])
    return w_avg


def sigmoid(x):
    return 1/(1+ math.exp(-x))