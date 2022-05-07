import torch
from model.baseline import CNNRegressor
from manager import Manager
import pandas as pd

CFG = {
    'IMG_SIZE': 128,
    'EPOCHS': 10,
    'LEARNING_RATE': 2e-3,
    'BATCH_SIZE': 1,
    'SEED': 41,
    'trainpath': './dataset/train',
    'testpath': './dataset/test'
}


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ## chage name!
    neural_engine  = torch.load('./saved/best_entire_model_100.pth')
    manager = Manager(neural_engine, CFG, device)

    preds = manager.predict()
    
    submission = pd.read_csv('./dataset/sample_submission.csv')
    submission['leaf_weight'] = preds
    submission.to_csv('./result.csv', index=False)
            


