"""Trainer for DeepPhys."""
import copy
import logging
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
# from evaluation.metrics import calculate_metrics
# from neural_methods.loss.NegPearsonLoss import Neg_Pearson
# from neural_methods.model.DeepPhys import DeepPhys
# from neural_methods.trainer.BaseTrainer import BaseTrainer
import wandb
from tqdm import tqdm
from models.Fed import sigmoid
from rPPG_file.evaluation.metrics import calculate_metrics, Kurtosis_calculate_metrics
from rPPG_file.neural_methods.model.DeepPhys import DeepPhys
from rPPG_file.neural_methods.trainer.BaseTrainer import BaseTrainer


class DeepPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = DeepPhys(img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS*config.TRAIN.num_users, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")

    def Fed_train(self, data_loader,epotch,state,i,j):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        self.model.load_state_dict(state)
        self.model.train()
        mean_training_losses = []
        mean_valid_losses = []
        # lrs = []
        for epoch in range(epotch):
            # print("==== 第 %s个用户 Training Epoch: %s ===="%(i,epoch))
            running_loss = 0.0
            train_loss = []
            # self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("全局第%s轮:第%s个用户:Train epoch %s" %(j+1,i+1,epoch+1))
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                # Append the current learning rate to the list
                # lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                # self.scheduler.step()
                train_loss.append(loss.item())
                # print(loss,lrs)
                tbar.set_postfix({"loss": loss.item()})
            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            # self.save_model(epoch)
            # if not self.config.TEST.USE_LAST_EPOCH:
            #     valid_loss = self.valid(data_loader)
            #     mean_valid_losses.append(valid_loss)
            _,MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg=self.Fed_test(data_loader, copy.deepcopy(self.model.state_dict()))
            wandb.log({'%s用户的epoch'%(i+1): (j*epotch+epoch+1),'%s用户的MAE'%(i+1): MAE, '%s用户的RMSE'%(i+1): RMSE, '%s用户的MAPE'%(i+1): MAPE,
                       '%s用户的correlation_coefficient'%(i+1): correlation_coefficient, '%s用户的SNR'%(i+1): SNR, '%s用户的MACC_avg'%(i+1): MACC_avg})
                # print('validation loss: ', valid_loss)
                # if self.min_valid_loss is None:
                #     self.min_valid_loss = valid_loss
                #     self.best_epoch = epoch
                #     print("Update best model! Best epoch: {}".format(self.best_epoch))
                # elif (valid_loss < self.min_valid_loss):
                #     self.min_valid_loss = valid_loss
                #     self.best_epoch = epoch
                #     print("Update best model! Best epoch: {}".format(self.best_epoch))
        # if not self.config.TEST.USE_LAST_EPOCH:
        #     print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        # if self.config.TRAIN.PLOT_LOSSES_AND_LR:
        #     self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)
        return copy.deepcopy(self.model.state_dict()),mean_training_losses,mean_valid_losses

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)
    def valid_global(self, data_loader,state_dict):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        self.model.load_state_dict(state_dict)
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)
    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = self.model(data_test)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def Fed_test(self, data_loader,state):
        predictions = dict()
        labels = dict()
        self.model.load_state_dict(state)
        self.model.eval()
        valid_loss = []
        print("Running model evaluation on the global model!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["valid"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = self.model(data_test)
                loss = self.criterion(pred_ppg_test, labels_test)
                valid_loss.append(loss.item())
                labels_test = labels_test.cpu()
                pred_ppg_test = pred_ppg_test.cpu()
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg=calculate_metrics(predictions, labels, self.config)
        valid_loss = np.asarray(valid_loss)
        # if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs
        #     self.save_test_outputs(predictions, labels, self.config)
        return np.mean(valid_loss),MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg
    def save_model(self, index):
        """Inits parameters from args and the writer for TensorboardX."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
    def model_initial(self):
        return self.model
    def all_train(self, data_loader_all,epotch,state,j,config,global_datadict):
        self.model.load_state_dict(state)
        self.model.train()
        mean_training_losses = []
        mean_valid_losses = []
        for epoch in range(epotch):
            # print("==== 第 %s个用户 Training Epoch: %s ===="%(i,epoch))
            running_loss = 0.0
            train_loss = []
            for data_loader in data_loader_all:
                for idx, batch in enumerate(data_loader['train']):
                    data, labels = batch[0].to(
                        self.device), batch[1].to(self.device)
                    N, D, C, H, W = data.shape
                    data = data.view(N * D, C, H, W)
                    labels = labels.view(-1, 1)
                    self.optimizer.zero_grad()
                    pred_ppg = self.model(data)
                    loss = self.criterion(pred_ppg, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss.append(loss.item())
                mean_training_losses.append(np.mean(train_loss))
            for k in range(0,3):
                _,MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg=self.Fed_test(data_loader_all[k], copy.deepcopy(self.model.state_dict()))
                wandb.log({'%s数据集 epoch'%config.TRAIN.dataset_list[k]: (j*epotch+epoch+1),'%s数据集 MAE'%config.TRAIN.dataset_list[k]: MAE, '%s数据集 RMSE'%config.TRAIN.dataset_list[k]: RMSE, '%s数据集 MAPE'%config.TRAIN.dataset_list[k]: MAPE,
                           '%s数据集 correlation_coefficient'%config.TRAIN.dataset_list[k]: correlation_coefficient, '%s数据集 SNR'%config.TRAIN.dataset_list[k]: SNR, '%s数据集 MACC_avg'%config.TRAIN.dataset_list[k]: MACC_avg})
            _, MAE, RMSE, MAPE, correlation_coefficient, SNR, MACC_avg = self.Fed_test(global_datadict,
                                                                                                copy.deepcopy(self.model.state_dict()))
            # print(MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg)
            wandb.log({'UBFC-PHYS epoch': (epoch + 1), 'UBFC-PHYS MAE': MAE,
                       'UBFC-PHYS RMSE': RMSE, 'UBFC-PHYS MAPE': MAPE,
                       'UBFC-PHYS correlation_coefficient': correlation_coefficient,
                       'UBFC-PHYS SNR': SNR,
                       'UBFC-PHYS MACC_avg': MACC_avg})
        return copy.deepcopy(self.model.state_dict()),mean_training_losses,mean_valid_losses
    def Fed_intra_train(self, data_loader, epotch, state, i, j,config,datadict_list,global_datadict):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        self.model.load_state_dict(state)
        self.model.train()
        mean_training_losses = []
        mean_valid_losses = []
        # lrs = []
        for epoch in range(epotch):
            train_loss = []
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("全局第%s轮:第%s个用户:Train epoch %s" % (j + 1, i + 1, epoch + 1))
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item()})
            mean_training_losses.append(np.mean(train_loss))
            for k in range(config.TRAIN.num_users):
                _, MAE, RMSE, MAPE, correlation_coefficient, SNR, MACC_avg = self.Fed_test(datadict_list[k], copy.deepcopy(
                    self.model.state_dict()))
                wandb.log({'%s用户的 %s epoch' % ((i + 1),config.TRAIN.dataset_list[k]): (j * epotch + epoch + 1), '%s用户的 %s MAE' % ((i + 1),config.TRAIN.dataset_list[k]): MAE,
                           '%s用户的 %s RMSE' % ((i + 1),config.TRAIN.dataset_list[k]): RMSE, '%s用户的 %s MAPE' % ((i + 1),config.TRAIN.dataset_list[k]): MAPE,
                           '%s用户的 %s correlation_coefficient' % ((i + 1),config.TRAIN.dataset_list[k]): correlation_coefficient, '%s用户的 %s SNR' % ((i + 1),config.TRAIN.dataset_list[k]): SNR,
                           '%s用户的 %s MACC_avg' % ((i + 1),config.TRAIN.dataset_list[k]): MACC_avg})
            _, MAE, RMSE, MAPE, correlation_coefficient, SNR, MACC_avg = self.Fed_test(global_datadict,
                                                                                       copy.deepcopy(
                                                                                           self.model.state_dict()))
            # print(MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg)
            wandb.log({'%s用户的 UBFC-PHYS epoch'% (i + 1): (j * epotch + epoch + 1), '%s用户的 UBFC-PHYS MAE'% (i + 1): MAE,
                       '%s用户的 UBFC-PHYS RMSE'% (i + 1): RMSE, '%s用户的 UBFC-PHYS MAPE'% (i + 1): MAPE,
                       '%s用户的 UBFC-PHYS correlation_coefficient'% (i + 1): correlation_coefficient,
                       '%s用户的 UBFC-PHYS SNR'% (i + 1): SNR,
                       '%s用户的 UBFC-PHYS MACC_avg'% (i + 1): MACC_avg})
        return copy.deepcopy(self.model.state_dict()), mean_training_losses, mean_valid_losses
    def SNR_Fed_intra_train(self, data_loader, epotch, state, i, j,config,datadict_list,global_datadict,num):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        self.model.load_state_dict(state)
        self.model.train()
        mean_training_losses = []
        mean_valid_losses = []
        # lrs = []
        for epoch in range(epotch):
            train_loss = []
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("全局第%s轮:第%s个用户:Train epoch %s" % (j + 1, i + 1, epoch + 1))
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item()})
            mean_training_losses.append(np.mean(train_loss))
            for k in range(config.TRAIN.num_users):
                if (k == i):
                    _, MAE, RMSE, MAPE, correlation_coefficient, SNR, MACC_avg,Kurtosis,SNR2 = self.new_Fed_test(datadict_list[k], copy.deepcopy(
                    self.model.state_dict()))
                # wandb.log({'%s用户的 %s epoch' % ((i + 1),config.TRAIN.dataset_list[k]): (j * epotch + epoch + 1), '%s用户的 %s MAE' % ((i + 1),config.TRAIN.dataset_list[k]): MAE,
                #            '%s用户的 %s RMSE' % ((i + 1),config.TRAIN.dataset_list[k]): RMSE, '%s用户的 %s MAPE' % ((i + 1),config.TRAIN.dataset_list[k]): MAPE,
                #            '%s用户的 %s correlation_coefficient' % ((i + 1),config.TRAIN.dataset_list[k]): correlation_coefficient, '%s用户的 %s SNR' % ((i + 1),config.TRAIN.dataset_list[k]): SNR,
                #            '%s用户的 %s MACC_avg' % ((i + 1),config.TRAIN.dataset_list[k]): MACC_avg})

                    #value=1-np.tanh(Kurtosis)**2
                    #value=sigmoid(SNR)
                    # value=sigmoid(SNR)+1.1*num[i]/sum(num)
                    value=sigmoid(num[i])
        return copy.deepcopy(self.model.state_dict()), mean_training_losses, mean_valid_losses,value
    def new_Fed_test(self, data_loader,state):
        predictions = dict()
        labels = dict()
        self.model.load_state_dict(state)
        self.model.eval()
        valid_loss = []
        print("Running model evaluation on the global model!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["valid"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = self.model(data_test)
                loss = self.criterion(pred_ppg_test, labels_test)
                valid_loss.append(loss.item())
                labels_test = labels_test.cpu()
                pred_ppg_test = pred_ppg_test.cpu()
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg,Kurtosis,SNR2=Kurtosis_calculate_metrics(predictions, labels, self.config)
        valid_loss = np.asarray(valid_loss)
        # if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs
        #     self.save_test_outputs(predictions, labels, self.config)
        return np.mean(valid_loss),MAE,RMSE,MAPE,correlation_coefficient,SNR,MACC_avg,Kurtosis,SNR2