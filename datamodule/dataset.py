from torch.utils.data import Dataset
import cv2
import pandas as pd

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
        
        meta_data.plot.box()
        exit()
        # for column in meta_data:
        #     print(column)

        # print(meta_data)
        # exit()
        

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image




    def __len__(self):
        return len(self.img_path_list)
