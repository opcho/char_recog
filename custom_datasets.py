from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class Chars74k(Dataset):

    def __init__(self, csv_file, root_dir,CLASSES, target_transform,transform=None,transform2=None):

        self.data_info = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.CLASSES=CLASSES
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        assert os.path.isfile(csv_file)
        for i in range(len(self.data_info.index)):
            assert os.path.exists(os.path.join(self.root_dir, self.data_info.iloc[i, 0]))

    def __len__(self):
        return len(self.data_info.index)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        # label to index number
        label=self.data_info.iloc[idx, 1]

        image1 = self.transform(image)
        image2 = self.transform2(image)

        if self.target_transform:
            label = self.target_transform(self.CLASSES,label)

        return image1,image2,label

if __name__ == "__main__":
    # Call dataset
    pass
