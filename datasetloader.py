#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# import os
# import sys
#
# print(os.path.join(sys.prefix, 'Library', 'bin', 'geos_c.dll'))
import argparse
import random

import matplotlib
from torch.utils.data import DataLoader

from config import get_config
from rPPG_file import dataset
from rPPG_file.dataset import data_loader
from rPPG_file.neural_methods import trainer
from utils.rPPG_options import args_parser_rppg

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser, add_args
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def initial():
    # parse args
    # args = args_parser_rppg()
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()
    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')
    train_loader = []
    valid_loader = []
    # load dataset and split users
    for data_set in config.TRAIN.dataset_list:
        if data_set == "UBFC-rPPG":
            train_loader.append(data_loader.UBFCrPPGLoader.UBFCrPPGLoader)
        elif data_set == "PURE":
            train_loader.append(data_loader.PURELoader.PURELoader)
        elif data_set == "SCAMPS":
            train_loader.append(data_loader.SCAMPSLoader.SCAMPSLoader)
        elif data_set == "MMPD":
            train_loader.append(data_loader.MMPDLoader.MMPDLoader)
        elif data_set == "BP4DPlus":
            train_loader.append(data_loader.BP4DPlusLoader.BP4DPlusLoader)
        elif data_set == "BP4DPlusBigSmall":
            train_loader.append(data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader)
        elif data_set == "UBFC-PHYS":
            train_loader.append(data_loader.UBFCPHYSLoader.UBFCPHYSLoader)
        elif data_set == "iBVP":
            train_loader.append(data_loader.iBVPLoader.iBVPLoader)
        elif data_set == "LGI_PPGI":
            train_loader.append(data_loader.LGI_PPGILoader.LGI_PPGILoader)
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")
    # print(config.VALID.DATA.valdataset_list)
    for valdata_set in config.VALID.DATA.valdataset_list:
        # print(valdata_set)
        if valdata_set == "UBFC-rPPG":
            valid_loader.append(data_loader.UBFCrPPGLoader.UBFCrPPGLoader)
        elif valdata_set == "PURE":
            valid_loader.append(data_loader.PURELoader.PURELoader)
        elif valdata_set == "SCAMPS":
            valid_loader.append(data_loader.SCAMPSLoader.SCAMPSLoader)
        elif valdata_set == "MMPD":
            valid_loader.append(data_loader.MMPDLoader.MMPDLoader)
        elif valdata_set == "BP4DPlus":
            valid_loader.append(data_loader.BP4DPlusLoader.BP4DPlusLoader)
        elif valdata_set == "BP4DPlusBigSmall":
            valid_loader.append(data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader)
        elif valdata_set == "UBFC-PHYS":
            valid_loader.append(data_loader.UBFCPHYSLoader.UBFCPHYSLoader)
        elif valdata_set == "iBVP":
            valid_loader.append(data_loader.iBVPLoader.iBVPLoader)
        elif valdata_set == "LGI_PPGI":
            valid_loader.append(data_loader.LGI_PPGILoader.LGI_PPGILoader)
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                                     SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP")

        # 加载数据
        # print(train_loader,valid_loader)
    datadict_list = []
    if (len(config.TRAIN.datapath_list) and len(config.TRAIN.dataset_list)):
        for i in range(config.TRAIN.num_users):
            data_loader_dict = dict()
            train_data_loader = train_loader[i](
                name="train",
                data_path=config.TRAIN.datapath_list[i],
                config_data=config.TRAIN.DATA,i=i)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=1,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
            valid_data = valid_loader[i](
                name="valid",
                data_path=config.VALID.DATA.valdatapath_list[i],
                config_data=config.VALID.DATA,i=i)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=1,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
            datadict_list.append(data_loader_dict)
    else:
        datadict_list =[None]
    return datadict_list,config
def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    return model_trainer.model_initial(),model_trainer