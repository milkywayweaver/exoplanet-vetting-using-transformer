import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class LightCurveDataset(Dataset):
    def __init__(self,data:pd.DataFrame,labels:pd.Series,transform:bool=True):
        self.data = data.to_numpy() #.copy()
        self.labels = labels.to_numpy() #.copy()
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            # 50% chance of x-axis flip
            p = np.random.uniform(0,1,1)
            if p >= 0.5:
                sample[:2001] = sample[:2001][::-1]
                sample[2001:4002] = sample[2001:4002][::-1]
                sample[4002:4203] = sample[4002:4203][::-1]
                sample[4203:4404] = sample[4203:4404][::-1]
        return sample,label