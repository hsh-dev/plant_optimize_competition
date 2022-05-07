import torch
import os
import numpy as np
import random

from model.baseline import CNNRegressor
from manager import Manager
from datamodule.data_controller import DataController

CFG = {
    'IMG_SIZE': 128,
    'EPOCHS': 10,
    'LEARNING_RATE': 2e-3,
    'BATCH_SIZE': 128,
    'VALID_BATCH_SIZE': 128,
    'TEST_BATCH_SIZE' : 1,
    'SEED': 41,
    'trainpath': './dataset/train',
    'testpath': './dataset/test',
    'feature': ['내부온도관측치', '외부온도관측치', '내부습도관측치', '외부습도관측치', 'CO2관측치', 'EC관측치', 
                '최근분무량', '화이트 LED동작강도', '레드 LED동작강도', '블루 LED동작강도', 
                '냉방온도', '냉방부하', '난방온도', '총추정광량', '백색광추정광량', '적색광추정광량', '청색광추정광량']
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything(CFG['SEED']) # Seed 고정

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("--- MODEL INIT ---")
    neural_engine  = CNNRegressor()
    
    data_manager = DataController(CFG)

    train_loader =  data_manager.train_loader

    for img, label in iter(train_loader):
        print(img)
        print(label)
        exit()





    print("--- MANAGER INIT ---")
    train_manager = Manager(neural_engine, CFG, data_manager, device)
    
    print("--- TRAIN START ---")
    train_manager.train(100)
        




