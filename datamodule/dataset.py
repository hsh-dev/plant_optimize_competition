from torch.utils.data import Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_path_list, meta_path_list, label_list, train_mode=True, transforms=None, calib_mode = False):
        self.transforms = transforms
        self.train_mode = train_mode
        self.calib_mode = calib_mode
        
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
        meta_data = meta_data.drop(labels='백색광추정광량', axis = 1)
        meta_data = meta_data.drop(labels='적색광추정광량', axis = 1)
        meta_data = meta_data.drop(labels='청색광추정광량', axis = 1)
     
        mean_data = meta_data.mean(axis=0, skipna=True)
        max_data = meta_data.max(axis=0, skipna=True)
        min_data = meta_data.min(axis=0, skipna=True)

        mean_data_np = np.array([mean_data.values], dtype=np.float16)
        max_data_np = np.array([max_data.values],dtype=np.float16)
        min_data_np = np.array([min_data.values], dtype=np.float16)

        meta = np.concatenate((mean_data_np, max_data_np), axis = 0)
        meta = np.concatenate((meta, min_data_np), axis = 0)
        meta = torch.Tensor(meta.reshape(-1))
        meta = torch.log(meta + 1)
        meta = torch.sigmoid(meta - 2)

        if self.calib_mode:
            if self.train_mode:
                label = self.label_list[index]
                return image, label, meta
            else:
                return image, meta
        else:
            if self.train_mode:
                label = self.label_list[index]
                return image, label
            else:
                return image


    def __len__(self):
        return len(self.img_path_list)
