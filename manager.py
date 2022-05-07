import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm
import numpy as np

from datamodule.data_controller import DataController

class Manager():
    def __init__(self, model, datamanager, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.datamanager = datamanager

        self.optimizer = torch.optim.SGD(params = model.parameters(), lr = config["LEARNING_RATE"])
        self.train_loader = self.datamanager.get_dataloader('train')
        self.valid_loader = self.datamanager.get_dataloader('valid')
        self.test_loader = self.datamanager.get_dataloader('test')
        
    def train(self, epochs):
        self.model.to(self.device)
        scheduler = None
        
        # Loss Function
        criterion = nn.L1Loss().to(self.device)
        best_mae = 9999
        
        for epoch in range(1,epochs+1):
            self.model.train()
            train_loss = []
            for img, label in tqdm(iter(self.train_loader)):
                img, label = img.float().to(self.device), label.float().to(self.device)
                
                self.optimizer.zero_grad()

                # Data -> Model -> Output
                logit = self.model(img)
                # Calc loss
                loss = criterion(logit.squeeze(1), label)

                # backpropagation
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                
            if scheduler is not None:
                scheduler.step()
                
            # Evaluation Validation set
            vali_mae = self.validation(criterion)
            
            print(f'Epoch [{epoch}] Train MAE : [{np.mean(train_loss):.5f}] Validation MAE : [{vali_mae:.5f}]\n')
            
            # Model Saved
            if best_mae > vali_mae:
                best_mae = vali_mae
                filename = './saved/best_entire_model_' + str(epoch) + '.pth'
                torch.save(self.model, filename)
                print('Model Saved.')
        
        
    def validation(self, criterion):
        self.model.eval() # Evaluation
        vali_loss = []
        with torch.no_grad():
            for img, label in tqdm(iter(self.valid_loader)):
                img, label = img.float().to(self.device), label.float().to(self.device)

                logit = self.model(img)
                loss = criterion(logit.squeeze(1), label)
                
                vali_loss.append(loss.item())

        vali_mae_loss = np.mean(vali_loss)
        return vali_mae_loss
    
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