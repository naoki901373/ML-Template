class TestDataset:
    
    def __init__(self, df):
        self.data = df.data.to_list()
        self.root = root
        self.preprocess = Preprocess()
        self.postprocess = Postprocess()
        self.augmentation = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        x = Image.open(os.path.join(self.root, self.data[i]))
        x = self.preprocess(x)
        if self.augmentation is not None:
            x = self.augmentation(x)
        x = self.postprocess(x)
        return {
            'x': x,
        }
    
class ValidDataset(TestDataset):
    
    def __init__(self, df):
        super().__init__(df)
        self.label = df.label.to_list()
        
    def __getitem__(self, i):
        ret = super().__getitem__(i)
        ret['target'] = self.label[i]
        return ret
    
class TrainDataset(ValidDataset):
    
    def __init__(self, df, augmentation):
        super().__init__(df)
        self.augmentation = augmentation