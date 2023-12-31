import collections
import pandas as pd
import torch
import os
import torchvision
from torch.utils.data import Dataset


class SchoolEqDataset(Dataset):
    def __init__(self, imgs_paths: collections.abc.Sequence[str], imgs_info_csv_path: str):
        super(SchoolEqDataset, self).__init__()

        images_info_csv = pd.read_csv(imgs_info_csv_path, delimiter=';')
        images_info = images_info_csv.set_index('Name').to_dict()
        labels_dict = images_info['Label']

        images = []
        self.labels = []
        for img_path in imgs_paths:
            # get the simplified path of the image:
            img_path_split = os.path.normpath(img_path).split(os.sep)
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            # remove images without labels
            if img_name in labels_dict:
                label = labels_dict[img_name]
            else:
                continue
            # append rest to the images list
            images.append(torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.GRAY))
            # append label
            self.labels.append(label)

        self.labels = torch.Tensor(self.labels).long()
        self.inputs = torch.stack(images).type(torch.uint8)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx, ...], self.labels[idx]

