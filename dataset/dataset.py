from torch.utils.data import Dataset
from torchvision import transforms as trans
from pathlib import Path
import pandas as pd
from PIL import  Image

class Dogs(Dataset):
    def __init__(self,img_folder,df_train,df_test,is_train=True,resize=True,transforms=None):
        self.root_folder = img_folder
        self.labels_list = list(df_test.columns)
        self.imgs = [img_path for img_path in self.root_folder.iterdir()]
        if is_train:
        	img_list = []
        	for img_path in self.imgs:
        		if img_path.name.split('.')[0] in list(df_train.index):
        			img_list.append(img_path)
        	self.imgs = img_list
        self.total_num = len(self.imgs)
        self.class_2_idx = {self.labels_list[i]:i for i in range(len(self.labels_list))}
        self.idx_2_class = {v:k for k,v in self.class_2_idx.items()}
        self.is_train = is_train
        self.test_marks = df_test.index
        if self.is_train:
            self.train_targets = [self.class_2_idx[df_train.loc[self.imgs[i].name.split('.')[0]]['breed']] 
                                  for i in range(self.total_num)]
        if transforms is None:
            if resize == False:
                self.transforms = trans.Compose([
                    trans.Resize(360),
                    trans.CenterCrop(360),
                    trans.ToTensor(),
                    ])
            else:
                self.transforms = trans.Compose([
                    trans.Resize((360,360)),
                    trans.ToTensor(),
                    ]) 
        else:
            self.transforms = transforms
    def __getitem__(self,index):
        img_path = self.imgs[index]
        if self.is_train:
            label = self.train_targets[index]
        else:
            label = self.test_marks[index]         
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label
    
    def __len__(self):
        return self.total_num        