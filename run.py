import torch
import os
import numpy as np
import random
import argparse
import neptune.new as neptune

from model.baseline import CNNRegressor
from manager import Manager
from datamodule.data_controller import DataController
from model.resnet import ResNet101

CFG = {
    'IMG_SIZE': 128,
    'EPOCHS': 300,
    'LEARNING_RATE': 1e-3,
    'MIN_LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 64,
    'VALID_BATCH_SIZE': 64,
    'TEST_BATCH_SIZE' : 1,
    'SEED': 41,
    'trainpath': './dataset/train',
    'testpath': './dataset/test',
    'feature': ['내부온도관측치', '외부온도관측치', '내부습도관측치', '외부습도관측치', 'CO2관측치', 'EC관측치', 
                '최근분무량', '화이트 LED동작강도', '레드 LED동작강도', '블루 LED동작강도', 
                '냉방온도', '냉방부하', '난방온도', '총추정광량', '백색광추정광량', '적색광추정광량', '청색광추정광량'],
    'empty_data': ['CASE08', 'CASE09', 'CASE22','CASE23', 'CASE26', 'CASE30', 'CASE31', 'CASE49', 'CASE59', 'CASE71', 'CASE72','CASE73'],
    'empty_data_subject': ['CASE60_20', 'CASE60_29', 'CASE60_33', 'CASE60_32', 'CASE60_26', 'CASE60_23', 
                           'CASE52_01', 'CASE02_10', 'CASE60_29', 'CASE70_23', 'CASE02_11', 'CASE34_01', 
                           'CASE56_01', 'CASE60_21', 'CASE60_27', 'CASE70_24', 'CASE70_20', 'CASE60_34',
                           'CASE60_28', 'CASE70_21', 'CASE40_01', 'CASE60_31', 'CASE63_01', 'CASE70_22',
                           'CASE60_22', 'CASE60_30', 'CASE60_24', 'CASE70_19', 'CASE60_25', 'CASE44_01']
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", default="./output")
    parser.add_argument("-n", "--experiment_name", default="new_test")
    parser.add_argument(
        "--enable_log",
        help="Decide whether to upload log on neptune or not",
        action='store_true'
    )
    args = parser.parse_args()
    output_path = os.path.abspath(args.save_path)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    CFG["save_path"] = os.path.join(args.save_path, args.experiment_name)
    if not os.path.isdir(CFG["save_path"]):
        os.mkdir(CFG["save_path"])
        
    neptune_callback = None
    if args.enable_log:
        neptune_callback = neptune.init(
            name=args.experiment_name,
            project="hsh-dev/jeff",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDNlMTkxZS0wOTE4LTRhYzUtODUzNS1hNGUyOTkzMTU0MjgifQ==",
            source_files=["run.py","manager.py","submit.py"
                          "./datamodel/*",
                          "./model/*"]
        )
        neptune_callback["parameters"] = CFG
    
    # Seed
    seed_everything(CFG['SEED'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("--- MODEL INIT ---")
    # neural_engine  = CNNRegressor()
    neural_engine = ResNet101()

    data_manager = DataController(CFG)

    print("--- MANAGER INIT ---")
    train_manager = Manager(neural_engine, data_manager, CFG, device, args.enable_log, neptune_callback)
    
    print("--- TRAIN START ---")
    train_manager.train()
        




