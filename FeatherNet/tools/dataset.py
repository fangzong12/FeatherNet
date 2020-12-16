from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import cv2


class FaceQualityDataset(Dataset):
    def __init__(self, root='', ann_file='', transform=None):

        self.transform = transform
        f_read = open(ann_file, 'r', encoding='utf-8')
        lines = f_read.readlines()
        self.labels = list()
        self.imgs = list()
        for line in lines:
            line = line.strip()
            items = line.split('/')
            la = 0
            if 'good' == items[-2]:
                la = 0
            elif 'bad' == items[-2]:
                la = 1
            else:
                continue
            img_path = os.path.join(root, line)

            try:
                from PIL import Image
                img = Image.open(img_path)
            except IOError:
                print(img_path)
                continue

            try:
                img = np.array(img, dtype=np.float32)
            except:
                print('corrupt img', img_path)
                continue

            self.labels.append(la)
            self.imgs.append(img_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]

        image = cv2.imread(path)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))
        sample = Image.fromarray(np.array(image)).convert('RGB')

        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target



