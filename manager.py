from cgitb import enable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
import sys
import time

from tqdm import tqdm
import numpy as np

from datamodule.data_controller import DataController
from callback.neptune_callback import NeptuneCallback

class Manager():
    def __init__(self, model, tail, datamanager, config, device, enable_log = False, callback = None, calib_head = None):
        self.model = model
        self.tail = tail

        self.config = config
        self.device = device
        self.datamanager = datamanager
        self.logs = {}
        self.enable_log = enable_log
        self.scheduler = None

        self.use_calib = False
        
        self.calib_start_epoch = self.config["calib_start_epoch"]

        self.calib_mode_on = False
        self.calib_head = None
        # self.calib_tail = None

        if calib_head is not None:
            self.calib_head = calib_head
        # if calib_tail is not None:
        #     self.calib_tail = calib_tail
        # if (calib_head is not None) and (calib_tail is not None):
        #     self.use_calib = True
        if calib_head is not None:
            self.use_calib = True

        # initialize
        self.init_optimizer()
        self.init_data_loader()
        self.init_loss()
        if self.enable_log:
            self.init_callback(callback)

    def init_data_loader(self):
        self.train_loader = self.datamanager.get_dataloader('train')
        self.valid_loader = self.datamanager.get_dataloader('valid')
        self.test_loader = self.datamanager.get_dataloader('test')
    
    def init_optimizer(self):
        # self.optimizer = torch.optim.SGD(
        #     params=self.model.parameters(), lr=self.config["LEARNING_RATE"])
        self.optimizer = torch.optim.Adam([
            {'params' : self.model.parameters()},
            {'params' : self.tail.parameters()}],
            lr = self.config["LEARNING_RATE"], betas=(0.99, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= 20, eta_min = self.config["MIN_LEARNING_RATE"])

        if self.use_calib:
            self.calib_optimizer = torch.optim.Adam([
                {'params' : self.calib_head.parameters()}],
                lr = self.config["CALIB_LEARNING_RATE"], betas=(0.99, 0.999))
            self.calib_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.calib_optimizer, T_max= 100, eta_min = self.config["MIN_LEARNING_RATE"])


    def init_loss(self):
        self.criterion = nn.L1Loss().to(self.device)
    
    def init_callback(self, callback):
        self.neptune_callback = NeptuneCallback(callback)

    def reset_dataset(self, empty_remove = False):
        # call once when you want to reset dataset with no empty data
        self.datamanager.init_dataset(empty_remove)
        self.train_loader = self.datamanager.get_dataloader('train')
        self.valid_loader = self.datamanager.get_dataloader('valid')
        self.test_loader = self.datamanager.get_dataloader('test')
    

    def train(self, stop_epochs = None):
        self.model.to(self.device)
        self.tail.to(self.device)

        if self.calib_mode_on:
            self.calib_head.to(self.device)
            # self.calib_tail.to(self.device)
        best_mae = 9999
        
        epochs = self.config["EPOCHS"]
        if stop_epochs is not None:
            epochs = stop_epochs            
        
        for epoch in range(1,epochs+1):
            if self.use_calib and (epoch == self.calib_start_epoch):
                self.calib_mode_on = True
                self.reset_dataset(True)
                print("Empty Data Removed!!")
                self.calib_head.to(self.device)
                # self.calib_tail.to(self.device)

            # Train Loop
            self.train_loop(epoch)
            
            # Evaluation Validation set
            self.valid_loop(epoch)
            
            print("Epoch : {} | Train MAE : {} | Train score : {} | Valid MAE : {} | Valid score : {} |"\
                .format(epoch, self.logs["train_mae"], self.logs["train_score"], self.logs["valid_mae"], self.logs["valid_score"]))
            
            
            # Model Saved
            if epoch % 20 == 0:
                self.save_model(epoch)
            if best_mae > self.logs["valid_mae"]:
                best_mae = self.logs["valid_mae"]
                if epoch > 100:
                    self.save_best_model(epoch)

            # Neptune Save
            self.logs["learning_rate"] = self.scheduler.get_last_lr()
            if self.enable_log:
                self.neptune_callback.save(self.logs)

            # Update LR
            if (self.scheduler is not None) and (epoch < 100):
                self.scheduler.step(epoch)
            
            if self.calib_mode_on:
                if epoch < 100:
                    self.calib_scheduler.step()

            # Update Train/Valid
            if epoch % 50 == 0:
                print("Dataset Update!!")
                if self.calib_mode_on:
                    self.reset_dataset(True)
                else:
                    self.reset_dataset(False)

    
    def train_loop(self, epoch):
        self.model.train()
        self.tail.train()

        if self.calib_mode_on:
            self.calib_head.train()
            # self.calib_tail.train()
            self.model.eval()
            self.tail.eval()

        total_train_loss = []
        tmp_train_loss = []
        
        total_error_list = []
        total_true_list = []
        tmp_error_list = []
        tmp_true_list = []
        
        steps = len(self.train_loader)
        step = 1
        prev_time = time.time()
        
        if self.calib_mode_on == False:
            for img, label in iter(self.train_loader):
                img, label = img.float().to(self.device), label.float().to(self.device)
                self.optimizer.zero_grad()

                model_output = self.model(img)
                logit = self.tail(model_output)
                loss = self.criterion(logit.squeeze(1), label)
                loss.backward()
                self.optimizer.step()
                
                tmp_train_loss.append(loss.item())
                total_train_loss.append(loss.item())
                
                logit_ = logit.cpu()
                label_ = label.cpu()
                error = np.abs(logit_.squeeze(1).detach().numpy()- label_.detach().numpy())
                tmp_error_list.extend(error)
                tmp_true_list.extend(np.abs(label_.detach().numpy()))
                total_error_list.extend(error)
                total_true_list.extend(np.abs(label_.detach().numpy()))
                
                if step % 2 == 0:
                    print("[{}]| STEP : ({}/{}) | TRAIN LOSS : {} | TRAIN SCORE : {} | Time : {}]".format(
                        epoch, step, steps, np.mean(tmp_train_loss), np.mean(tmp_error_list)/np.mean(tmp_true_list), time.time()-prev_time))
                    sys.stdout.flush()
                    
                    tmp_train_loss.clear()
                    tmp_error_list.clear()
                    tmp_true_list.clear()
                    prev_time = time.time()
                step = step+1
        else:
            for img, label, meta in iter(self.train_loader):
                img, label, meta = img.float().to(self.device), label.float().to(self.device), meta.float().to(self.device)

                self.optimizer.zero_grad()
                self.calib_optimizer.zero_grad()
                
                with torch.no_grad():
                    model_output = self.model(img)
                    tail_output = self.tail(model_output)
                    tail_output = torch.log(tail_output + 1)
                    tail_output = torch.sigmoid(tail_output - 2)

                calib_head_input = torch.cat((tail_output, meta), dim = 1)
                calib_head_output = self.calib_head(calib_head_input)

                # calib_tail_input = torch.cat((tail_output, calib_head_output), dim = 1)
                # logit = self.calib_tail(calib_tail_input)
                logit = calib_head_output
                loss = self.criterion(logit.squeeze(1), label)
                loss.backward()

                # self.optimizer.step()
                self.calib_optimizer.step()

                tmp_train_loss.append(loss.item())
                total_train_loss.append(loss.item())
                
                logit_ = logit.cpu()
                label_ = label.cpu()
                error = np.abs(logit_.squeeze(1).detach().numpy()- label_.detach().numpy())
                tmp_error_list.extend(error)
                tmp_true_list.extend(np.abs(label_.detach().numpy()))
                total_error_list.extend(error)
                total_true_list.extend(np.abs(label_.detach().numpy()))
                
                if step % 1 == 0:
                    print("[{}]| STEP : ({}/{}) | TRAIN LOSS : {} | TRAIN SCORE : {} | Time : {}]".format(
                        epoch, step, steps, np.mean(tmp_train_loss), np.mean(tmp_error_list)/np.mean(tmp_true_list), time.time()-prev_time))
                    sys.stdout.flush()
                    
                    tmp_train_loss.clear()
                    tmp_error_list.clear()
                    tmp_true_list.clear()
                    prev_time = time.time()
                step = step+1

        train_score = np.mean(total_error_list) / np.mean(total_true_list)
        train_mae_loss = np.mean(total_train_loss)
        
        self.logs["train_score"] = train_score
        self.logs["train_mae"] = train_mae_loss
         
    
    def valid_loop(self, epoch):
        self.model.eval() # Evaluation
        self.tail.eval()

        if self.calib_head:
            self.calib_head.eval()
            # self.calib_tail.eval()

        total_valid_loss = []
        tmp_valid_loss = []
        total_error_list = []
        total_true_list = []
        tmp_error_list = []
        tmp_true_list = []
        
        steps = len(self.valid_loader)
        step = 1
        prev_time = time.time()
        
        with torch.no_grad():
            if not self.calib_mode_on:
                for img, label in iter(self.valid_loader):
                    img, label = img.float().to(self.device), label.float().to(self.device)

                    model_output = self.model(img)
                    logit = self.tail(model_output)
                    loss = self.criterion(logit.squeeze(1), label)
                    
                    tmp_valid_loss.append(loss.item())
                    total_valid_loss.append(loss.item())

                    logit_ = logit.cpu()
                    label_ = label.cpu()
                    error = np.abs(logit_.squeeze(1).detach().numpy() -
                            label_.detach().numpy())
                    tmp_error_list.extend(error)
                    tmp_true_list.extend(np.abs(label_.detach().numpy()))
                    total_error_list.extend(error)
                    total_true_list.extend(np.abs(label_.detach().numpy()))
                    
                    if step % 2 == 0:
                        print("[{}]| STEP : ({}/{}) | VALID LOSS : {} | VALID SCORE : {} | Time : {}]".format(
                            epoch, step, steps, np.mean(tmp_valid_loss), np.mean(tmp_error_list)/np.mean(tmp_true_list), time.time()-prev_time))
                        sys.stdout.flush()

                        tmp_valid_loss.clear()
                        tmp_error_list.clear()
                        tmp_true_list.clear()
                        prev_time = time.time()
                    step = step+1
            
            else:
                for img, label, meta in iter(self.valid_loader):
                    img, label, meta = img.float().to(self.device), label.float().to(self.device), meta.float().to(self.device)

                    model_output = self.model(img)
                    tail_output = self.tail(model_output)
                    tail_output = torch.log(tail_output + 1)
                    tail_output = torch.sigmoid(tail_output - 2)

                    calib_head_input = torch.cat((tail_output, meta), dim = 1)
                    calib_head_output = self.calib_head(calib_head_input)

                    # calib_tail_input = torch.cat((tail_output, calib_head_output), dim = 1)
                    # logit = self.calib_tail(calib_tail_input)
                    logit = calib_head_output
                    loss = self.criterion(logit.squeeze(1), label)
                    tmp_valid_loss.append(loss.item())
                    total_valid_loss.append(loss.item())

                    logit_ = logit.cpu()
                    label_ = label.cpu()
                    error = np.abs(logit_.squeeze(1).detach().numpy() -
                            label_.detach().numpy())
                    tmp_error_list.extend(error)
                    tmp_true_list.extend(np.abs(label_.detach().numpy()))
                    total_error_list.extend(error)
                    total_true_list.extend(np.abs(label_.detach().numpy()))
                    
                    if step % 2 == 0:
                        print("[{}]| STEP : ({}/{}) | VALID LOSS : {} | VALID SCORE : {} | Time : {}]".format(
                            epoch, step, steps, np.mean(tmp_valid_loss), np.mean(tmp_error_list)/np.mean(tmp_true_list), time.time()-prev_time))
                        sys.stdout.flush()

                        tmp_valid_loss.clear()
                        tmp_error_list.clear()
                        tmp_true_list.clear()
                        prev_time = time.time()
                    step = step+1                

        valid_score = np.mean(total_error_list) / np.mean(total_true_list)
        valid_mae_loss = np.mean(total_valid_loss)

        self.logs["valid_score"] = valid_score
        self.logs["valid_mae"] = valid_mae_loss
        
    
    def predict(self):
        self.model.to(self.device)
        self.tail.to(self.device)
        self.calib_head.to(self.device)

        self.model.eval()
        self.tail.eval()
        self.calib_head.eval()

        model_pred = []
        with torch.no_grad():
            for img, meta in iter(self.test_loader):
                img, meta = img.float().to(self.device), meta.float().to(self.device)

                model_output = self.model(img)
                tail_output = self.tail(model_output)                    
                tail_output = torch.log(tail_output + 1)
                tail_output = torch.sigmoid(tail_output - 2)

                calib_input = torch.cat((tail_output, meta), dim = 1)
                logit = self.calib_head(calib_input)

                pred_logit = logit.squeeze(1).detach().cpu()

                model_pred.extend(pred_logit.tolist())
                
        return model_pred

    def save_model(self, epoch):
        filename = self.config["save_path"] + "/model_" + str(epoch) + '.pth'
        torch.save(self.model, filename)
        filename = self.config["save_path"] + "/tail_" + str(epoch) + '.pth'
        torch.save(self.tail, filename)            
        if self.calib_mode_on:
            filename = self.config["save_path"] + "/calib_head_" + str(epoch) + '.pth'
            torch.save(self.calib_head, filename)
            # filename = self.config["save_path"] + "/calib_tail_" + str(epoch) + '.pth'
            # torch.save(self.calib_tail, filename)       


    def save_best_model(self, epoch):
        filename = self.config["save_path"] + "/best_model_" +  str(epoch) + '.pth'
        torch.save(self.model, filename)
        filename = self.config["save_path"] + "/best_tail_" +  str(epoch) + '.pth'
        torch.save(self.tail, filename)  

        if self.calib_mode_on:
            filename = self.config["save_path"] + "/best_calib_head_" +  str(epoch) + '.pth'
            torch.save(self.calib_head, filename)
            # filename = self.config["save_path"] + "/best_calib_tail" + '.pth'
            # torch.save(self.calib_tail, filename)   