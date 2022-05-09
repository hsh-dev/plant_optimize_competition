import torch
from model.baseline import CNNRegressor
from manager import Manager
import pandas as pd
from datamodule.data_controller import DataController
from config import CFG


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ## chage name!
    model_name = 'res101_test'
    neural_engine  = torch.load('./output/' + str(model_name) + '/model_220.pth')
    data_manager = DataController(CFG)
    manager = Manager(neural_engine, data_manager, CFG, device)

    preds = manager.predict()
    
    submission = pd.read_csv('./dataset/sample_submission.csv')
    submission['leaf_weight'] = preds
    submission.to_csv('./' + str(model_name) + '_result.csv', index=False)
            


