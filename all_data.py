#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
from utils.datasetloader import initial, train_and_test
matplotlib.use('Agg')
import copy
from models.Fed import FedAvg
import wandb

if __name__ == '__main__':
    datadict_list,config,global_datadict=initial()

    net_glob,moedl_trainer=train_and_test(config,datadict_list[0])
    wandb_config = dict(
        architecture="FedAvg_Deephys",
        dataset_id="UBFC,MMPD,PURE",
    )

    wandb.init(
        project="test",
        config=config,
    )
    net_glob.train()
    w_glob= copy.deepcopy(net_glob.state_dict())
    local_state,mean_training_losses,mean_valid_losses=moedl_trainer.all_train(datadict_list,config.TRAIN.local_ep,copy.deepcopy(w_glob),0,config,global_datadict)
    wandb.finish()


