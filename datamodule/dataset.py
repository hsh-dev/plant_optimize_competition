from torch.utils.data import Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_path_list, meta_path_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode

        self.img_path_list = img_path_list
        self.meta_path_list = meta_path_list
        self.label_list = label_list


    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        # Get image data
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        meta_path =self.meta_path_list[index]
        meta_data = pd.read_csv(meta_path)
        meta_data = meta_data.drop(labels='시간', axis=1)
        
        mean_data = meta_data.mean(axis=0, skipna=True)
        # max_data = meta_data.max(axis=0, skipna=True)
        # min_data = meta_data.min(axis=0, skipna=True)
        
        # print(meta_path + " asserting")
        # is_nan = True
        # for data in mean_data:
        #     if not np.isnan(data):
        #         is_nan = False
        # if is_nan:
        #     print(meta_path + "is all nan")
                        

        
        # exit()
        
        # null_detect = meta_data.notnull().sum()
        
        # for idx, value in enumerate(null_detect):
        #     if idx > 0:
        #         if value > 0:
        #             print("NULL DETECTED..")
        #             print(meta_path)
    
             
        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image




    def __len__(self):
        return len(self.img_path_list)
