import torch


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, csv_path, folder_path, img_size, transform=None):
        df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.img_ids = list(df['Image'])
        for i, d in enumerate(self.image_ids):
            self.image)ids[i] = folder_path + '/' + id
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
        with Image.open(self.image_ids[index]) as image:
            inputs = transforms.functional.resize(image, self.img_size)

        inputs = transforms.ToTensor()(inputs)
        inputs = self.transform(inputs)
        
        return inputs, self.labels[index]

    def __len__(self):
        return self.l
