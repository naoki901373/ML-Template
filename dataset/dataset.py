from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torch

_transform =transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224,224))]
                                )
                                
class TorchDataset(Dataset):
    def __init__(self,df,transform = _transform):
        self.df = df
        self.transform = _transform

    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):
        self.img = self.df['image']
        self.label = self.label['label']
        if self.transform:
            self.img = self.transform(self.img)
            self.label = self.transform(self.label)
        return self.img, self.label