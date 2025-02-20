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
    datadict_list,config=initial()

    net_glob,moedl_trainer=train_and_test(config,datadict_list[0])
    wandb_config = dict(
        architecture="FedAvg_Deephys",
        dataset_id="UBFC,MMPD,PURE",
    )

    wandb.init(
        project="my_first_try",
        config=config,
    )
    net_glob.train()
    #global_data=global_validdataset(config)
    # copy weights
    train_loss=[[],[],[]]
    val_loss=[[],[],[]]
    w_glob= copy.deepcopy(net_glob.state_dict())
    # po=[]
    num_data=[]
    for data in datadict_list:
        num_data.append(len(data['train']))
    for i in range(config.TRAIN.global_ep):
        # state_n=[]
        state_local=[]
        for j in range(config.TRAIN.num_users):
            # po.append(w_glob)
            local_state,mean_training_losses,mean_valid_losses=moedl_trainer.Fed_train(datadict_list[j],config.TRAIN.local_ep,copy.deepcopy(w_glob),j,i)
            train_loss[j].extend(mean_training_losses)
            val_loss[j].extend(mean_training_losses)
            state_local.append(copy.deepcopy(local_state))
        w_glob = FedAvg(state_local,num_data)
        # print(po[i*3]==po[i*3+1],po[i*3]==po[i*3+2])
        for k in range(config.TRAIN.num_users):
          _,MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg=moedl_trainer.Fed_test(datadict_list[k],copy.deepcopy(w_glob))
          # print(MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg)
          wandb.log({'%s epoch'%config.TRAIN.dataset_list[k]: (i + 1), '%s MAE'%config.TRAIN.dataset_list[k]:MAE,'%s RMSE'%config.TRAIN.dataset_list[k]:RMSE,'%s MAPE'%config.TRAIN.dataset_list[k]:MAPE,'%s correlation_coefficient'%config.TRAIN.dataset_list[k]:correlation_coefficient,'%s SNR'%config.TRAIN.dataset_list[k]:SNR,'%s MACC_avg'%config.TRAIN.dataset_list[k]:MACC_avg})
    wandb.finish()