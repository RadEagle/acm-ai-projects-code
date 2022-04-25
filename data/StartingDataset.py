import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, csv_path, folder_path, img_size, transform=None):
        df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.img_ids = list(df['Image'])

        for i, id in enumerate(self.img_ids):
            self.img_ids[i] = folder_path + '/' + id

        self.labels = list(df["Id"])
        self.l = len(self.labels)
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.img_size[0], padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        pass

    def __getitem__(self, index):
        #inputs = torch.zeros([3, 224, 224])
        inputs = 0
        with Image.open(self.img_ids[index]).convert('RGB') as image:
            inputs = transforms.functional.resize(image, self.img_size)

        inputs = transforms.ToTensor()(inputs)

        # print(self.img_ids[index])
        # print(inputs.size(dim=0))
        
        inputs = self.transform(inputs)
        
        return inputs, self.labels[index]

    def __len__(self):
        return self.l
