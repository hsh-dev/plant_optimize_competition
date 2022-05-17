import torch
from model.baseline import CNNRegressor
from manager import Manager
import pandas as pd
from datamodule.data_controller import DataController
from config import CFG


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ## chage name!
    
    model_name = 'resnet_50_lr_test'
    model = torch.load('./output/' + str(model_name) + '/best_model.pth')
    tail = torch.load('./output/' + str(model_name) + '/best_tail.pth' )
    # calib_head = torch.load('./output/' + str(model_name) + '/best_calib_head.pth')

    # model  = torch.load('./output/' + str(model_name) + '/model_' + '260.pth')
    # tail = torch.load('./output/' + str(model_name) + '/tail_' + '260.pth')
    # calib_head = torch.load('./output' + str(model_name) + '/calib_head_' + '260.pth')
    data_manager = DataController(CFG)

    # if using calib data
    # data_manager = DataController(CFG, True)

    manager = Manager(model, tail, data_manager, CFG, device)
    # manager = Manager(model, tail, data_manager, CFG, device, calib_head)

    preds = manager.predict()
    
    submission = pd.read_csv('./dataset/sample_submission.csv')
    submission['leaf_weight'] = preds
    submission.to_csv('./' + str(model_name) + '_result.csv', index=False)
            


