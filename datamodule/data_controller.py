import os
from glob import glob
from sys import meta_path
from idna import valid_label_length
import pandas as pd
from torch import _test_serialization_subcmul
from datamodule.dataset import CustomDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import random

class DataController():
    def __init__ (self, config):
        self.config = config
    
        self.init_dataset(False)

        self.train_loader = self.get_dataloader('train')
        self.valid_loader = self.get_dataloader('valid')
        self.test_loader = self.get_dataloader('test')

    def init_dataset(self, empty_remove = False):
        all_img_path, all_meta_path, all_label = self.get_train_data(self.config['trainpath'], empty_remove)
        self.test_img_path, self.test_meta_path = self.get_test_data(self.config['testpath'])

        assert (len(all_img_path) == len(all_meta_path) and len(all_meta_path) == len(all_label)), "length different!"
        self.all_data_length = len(all_img_path)
        train_length = int(self.all_data_length * 0.85)

        all_img_path, all_meta_path, all_label  = self.random_shuffle(all_img_path, all_meta_path, all_label)

        train_img_path = all_img_path[:train_length]
        train_meta_path = all_meta_path[:train_length]
        train_label = all_label[:train_length]

        self.train_img_path, self.train_meta_path, self.train_label = self.random_shuffle(train_img_path, train_meta_path, train_label)

        valid_img_path = all_img_path[train_length:]
        valid_meta_path = all_meta_path[train_length:]
        valid_label = all_label[train_length:]

        self.valid_img_path, self.valid_meta_path, self.valid_label = self.random_shuffle(valid_img_path, valid_meta_path, valid_label)


    def random_shuffle(self, L1, L2, L3):
        data_zip = list(zip(L1, L2, L3))
        random.shuffle(data_zip)
        L1, L2, L3 = zip(*data_zip)

        return L1, L2, L3


    
    def get_dataloader(self, type):
        train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.config['IMG_SIZE'], self.config['IMG_SIZE'])),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])

        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.config['IMG_SIZE'], self.config['IMG_SIZE'])),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])        
        
        if type == 'train':
            train_dataset = CustomDataset(self.train_img_path, self.train_meta_path, self.train_label, train_mode=True, transforms=train_transform)
            train_loader = DataLoader(train_dataset, batch_size = self.config['BATCH_SIZE'], shuffle=True, num_workers=0)
            return train_loader
        elif type == 'valid':
            valid_dataset = CustomDataset(self.valid_img_path, self.valid_meta_path, self.valid_label, train_mode=True, transforms=test_transform)
            valid_loader = DataLoader(valid_dataset, batch_size = self.config['VALID_BATCH_SIZE'], shuffle=False, num_workers=0)
            return valid_loader
        else:
            test_dataset = CustomDataset(self.test_img_path, self.test_meta_path, None, train_mode=False, transforms=test_transform)
            test_loader = DataLoader(test_dataset, batch_size = self.config['TEST_BATCH_SIZE'], shuffle=False, num_workers=0)
            return test_loader
            

    def get_train_data(self, data_dir, empty_remove = False):
        img_path_list = []
        label_list = []
        meta_path_list = []

        for case_name in os.listdir(data_dir):
            current_path = os.path.join(data_dir, case_name)
            
            if empty_remove and (case_name in self.config['empty_data']):
                print("{} has empty meta data".format(case_name))
                continue
            
            if os.path.isdir(current_path):
                # get image path
                tmp_img_path_list = []
                tmp_meta_path_list = []
                tmp_label_list = []
                
                tmp_img_path_list.extend(
                    glob(os.path.join(current_path, 'image', '*.jpg')))
                tmp_img_path_list.extend(
                    glob(os.path.join(current_path, 'image', '*.png')))
                tmp_img_path_list.sort()
                
                # get label
                label_df = pd.read_csv(current_path+'/label.csv')
                tmp_label_list.extend(label_df['leaf_weight'])
                
                remove_idx = []
                
                for idx, image_name in enumerate(tmp_img_path_list):
                    name_sliced = image_name.split('/')[-1].split('.')[0]
                    if empty_remove and (name_sliced in self.config['empty_data_subject']):
                        print("{} has empty meta data".format(name_sliced))
                        remove_idx.append(idx)
                
                # remove subject
                tmp_img_path_list = [ele for idx, ele in enumerate(tmp_img_path_list) if idx not in remove_idx]
                tmp_label_list = [ele for idx, ele in enumerate(tmp_label_list) if idx not in remove_idx]

                
                # get meta path
                for img_path in tmp_img_path_list:
                    meta_path = img_path.replace('image', 'meta')
                    meta_path = meta_path.replace('jpg', 'csv')
                    meta_path = meta_path.replace('png','csv')
                    
                    assert os.path.isfile(meta_path), str("PATH : " + meta_path + " is wrong!")
                    tmp_meta_path_list.append(meta_path)
                
                img_path_list.extend(tmp_img_path_list)
                meta_path_list.extend(tmp_meta_path_list)
                label_list.extend(tmp_label_list)

        return img_path_list, meta_path_list, label_list

    def get_test_data(self, data_dir):
        # get image path
        img_path_list = glob(os.path.join(data_dir, 'image', '*.jpg'))
        img_path_list.extend(glob(os.path.join(data_dir, 'image', '*.png')))
        img_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        meta_path_list = []

        # get meta path
        for img_path in img_path_list:
            meta_path = img_path.replace('image', 'meta')
            meta_path = meta_path.replace('jpg', 'csv')
            meta_path = meta_path.replace('png','csv')
            
            assert os.path.isfile(meta_path), str("PATH : " + meta_path + " is wrong!")
            meta_path_list.append(meta_path)


        return img_path_list, meta_path_list