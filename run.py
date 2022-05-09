import torch
import os
import numpy as np
import random
import argparse
import neptune.new as neptune

from model.baseline import CNNRegressor
from manager import Manager
from datamodule.data_controller import DataController
from model.resnet import ResNet101, ResNet50, ResNetTail
from model.calibration import CalibrationHead, CalibrationTail
from config import CFG

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
            source_files=["run.py","manager.py","submit.py",
                          "./datamodule/*",
                          "./callback/*",
                          "./model/*",
                          ]
        )
        neptune_callback["parameters"] = CFG
    
    # Seed
    seed_everything(CFG['SEED'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("--- MODEL INIT ---")
    # neural_engine  = CNNRegressor()
    main_net = ResNet50()
    main_tail = ResNetTail()

    calib_head = CalibrationHead
    calib_tail = CalibrationTail

    data_manager = DataController(CFG)

    print("--- MANAGER INIT ---")
    train_manager = Manager(main_net, main_tail, data_manager, CFG, device, args.enable_log, neptune_callback)
    
    print("--- TRAIN START ---")
    train_manager.train()
        




