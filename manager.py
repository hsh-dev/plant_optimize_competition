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
    def __init__(self, model, datamanager, config, device, enable_log = False, callback = None):
        self.model = model
        self.config = config
        self.device = device
        self.datamanager = datamanager
        self.logs = {}
        self.enable_log = enable_log
        self.scheduler = None

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
        self.optimizer = torch.optim.Adam(params = self.model.parameters(),
            lr = self.config["LEARNING_RATE"], betas=(0.99, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= 20, eta_min = self.config["MIN_LEARNING_RATE"])


    def init_loss(self):
        self.criterion = nn.L1Loss().to(self.device)
    
    def init_callback(self, callback):
        self.neptune_callback = NeptuneCallback(callback)

    def init_empty_data_remove(self):
        self.datamanager.init_dataset(empty_remove = True)
        self.train_loader = self.datamanager.get_dataloader('train')
        self.valid_loader = self.datamanager.get_dataloader('valid')
        self.test_loader = self.datamanager.get_dataloader('test')
    

    def train(self, stop_epochs = None):
        self.model.to(self.device)
        best_mae = 9999
        
        epochs = self.config["EPOCHS"]
        if stop_epochs is not None:
            epochs = stop_epochs            
        
        for epoch in range(1,epochs+1):
            # if epoch == 100:
            #     self.init_empty_data_remove()
            #     print("Empty Data Removed!!")

            # Train Loop
            self.train_loop()
            
            # Evaluation Validation set
            self.valid_loop()
            
            print("Epoch : {} | Train MAE : {} | Train score : {} | Valid MAE : {} | Valid score : {} |"\
                .format(epoch, self.logs["train_mae"], self.logs["train_score"], self.logs["valid_mae"], self.logs["valid_score"]))
            
            # Neptune Save
            if self.enable_log:
                self.neptune_callback.save(self.logs)
            
            # Model Saved
            self.save_model(epoch)
            if best_mae > self.logs["valid_mae"]:
                best_mae = self.logs["valid_mae"]
                self.save_best_model(epoch)

            # Update LR    
            if self.scheduler is not None:
                self.scheduler.step()
    
    def train_loop(self):
        self.model.train()
        total_train_loss = []
        tmp_train_loss = []
        
        total_error_list = []
        total_true_list = []
        tmp_error_list = []
        tmp_true_list = []
        
        steps = len(self.train_loader)
        step = 1
        prev_time = time.time()
        
        
        for img, label in iter(self.train_loader):
            img, label = img.float().to(self.device), label.float().to(self.device)
            
            self.optimizer.zero_grad()

            logit = self.model(img)
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
                print("[STEP : ({}/{}) | TRAIN LOSS : {} | TRAIN SCORE : {} | Time : {}]".format(step,
                      steps, np.mean(tmp_train_loss), np.mean(tmp_error_list)/np.mean(tmp_true_list), time.time()-prev_time))
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
         
    
    def valid_loop(self):
        self.model.eval() # Evaluation
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
            for img, label in iter(self.valid_loader):
                img, label = img.float().to(self.device), label.float().to(self.device)

                logit = self.model(img)
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
                    print("[STEP : ({}/{}) | VALID LOSS : {} | VALID SCORE : {} | Time : {}]".format(step,
                        steps, np.mean(tmp_valid_loss), np.mean(tmp_error_list)/np.mean(tmp_true_list), time.time()-prev_time))
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
        self.model.eval()
        model_pred = []
        with torch.no_grad():
            for img in tqdm(iter(self.test_loader)):
                img = img.float().to(self.device)

                pred_logit = self.model(img)
                pred_logit = pred_logit.squeeze(1).detach().cpu()

                model_pred.extend(pred_logit.tolist())
        return model_pred

    def save_model(self, epoch):
        filename = self.config["save_path"] + "/model_" + str(epoch) + '.pth'
        torch.save(self.model, filename)
    
    def save_best_model(self, epoch):
        filename = self.config["save_path"] + "/best_model" + '.pth'
        torch.save(self.model, filename)