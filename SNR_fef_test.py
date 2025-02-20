#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import random
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
from utils.datasetloader import initial, train_and_test
matplotlib.use('Agg')
import copy
from models.Fed import FedAvg, sigmoid
import wandb

if __name__ == '__main__':
    datadict_list,config,global_datadict=initial()

    net_glob,moedl_trainer=train_and_test(config,datadict_list[0])
    wandb_config = dict(
        architecture="FedAvg_Deephys",
        dataset_id="UBFC,MMPD,PURE",
    )

    wandb.init(
        project="SNR_SIG_test",
        # name=" 1e-3 lamata=1.1",
        name="chrom SNR",
        config=config,
    )
    net_glob.train()
    #global_data=global_validdataset(config)
    # copy weights
    w_glob= copy.deepcopy(net_glob.state_dict())
    # po=[]
    # num=[342,741,514]
    num=[2.037272759,-8.843465235,0.857540343]
    for i in range(config.TRAIN.global_ep):
        state_local=[]
        num_data = []
        for j in range(config.TRAIN.num_users):
            # po.append(w_glob)
            local_state,mean_training_losses,mean_valid_losses,snr=moedl_trainer.SNR_Fed_intra_train(datadict_list[j],config.TRAIN.local_ep,copy.deepcopy(w_glob),j,i,config,datadict_list,global_datadict,num)
            num_data.append(snr)
            state_local.append(copy.deepcopy(local_state))
        w_glob = FedAvg(state_local,num_data)
        # print(po[i*3]==po[i*3+1],po[i*3]==po[i*3+2])
        for k in range(config.TRAIN.num_users):
          _,MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg=moedl_trainer.Fed_test(datadict_list[k],copy.deepcopy(w_glob))
          # print(MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg)
          wandb.log({'%s epoch'%config.TRAIN.dataset_list[k]: (i + 1), '%s MAE'%config.TRAIN.dataset_list[k]:MAE,'%s RMSE'%config.TRAIN.dataset_list[k]:RMSE,'%s MAPE'%config.TRAIN.dataset_list[k]:MAPE,'%s correlation_coefficient'%config.TRAIN.dataset_list[k]:correlation_coefficient,'%s SNR'%config.TRAIN.dataset_list[k]:SNR,'%s MACC_avg'%config.TRAIN.dataset_list[k]:MACC_avg})
        _, MAE, RMSE, MAPE, correlation_coefficient, SNR, MACC_avg = moedl_trainer.Fed_test(global_datadict,
                                                                                            copy.deepcopy(w_glob))
        # print(MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg)
        wandb.log({'UBFC-PHYS epoch': (i + 1), 'UBFC-PHYS MAE': MAE,
                   'UBFC-PHYS RMSE': RMSE, 'UBFC-PHYS MAPE' : MAPE,
                   'UBFC-PHYS correlation_coefficient' : correlation_coefficient,
                   'UBFC-PHYS SNR' : SNR,
                   'UBFC-PHYS MACC_avg' : MACC_avg})
    wandb.finish()