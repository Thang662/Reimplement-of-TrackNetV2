from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import cv2
from heatmap import gen_binary_map
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_pil_image as ToPILImage
from torch.utils.data import DataLoader
import os

class Tennis(Dataset):
    """
    """
    def __init__(self, root, train, frame_in, is_sequential, transform = None, train_games = [i for i in range(1, 9)], r = 2.5, w = 512, h = 288):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.frame_in = frame_in
        self.is_sequential = is_sequential
        self.games = train_games if train else [i for i in range(1, 11) if i not in train_games]
        self.data = self.load_data()
        self.r = r
        self.w = w
        self.h = h
        if self.transform is None:
            if self.train:
                # self.transform = A.Compose([
                #         A.Resize(height = self.h, width = self.w, p = 1), 
                #         A.RandomSizedCrop(min_max_height = (self.h * 0.5, self.h), height = self.h, width = self.w, p = 0.5),
                #         A.HorizontalFlip(p = 0.5),
                #         A.Rotate(limit = 40, p = 0.5),
                #         A.OneOf([
                #             A.HueSaturationValue(p = 0.5),
                #             A.RGBShift(p = 0.7)
                #         ], p = 1),
                #         A.RandomBrightnessContrast(p = 0.5),
                #         A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
                #         ToTensorV2()
                #     ],
                #     keypoint_params = A.KeypointParams(format = 'xy'),
                # )
                self.transform = A.Compose([
                        A.Resize(height = self.h, width = self.w, p = 1),
                        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
                        ToTensorV2()
                    ],
                    keypoint_params = A.KeypointParams(format = 'xy'),
                )
            else:
                self.transform = A.Compose([
                        A.Resize(height = self.h, width = self.w, p = 1),
                        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
                        ToTensorV2()
                    ],
                    keypoint_params = A.KeypointParams(format = 'xy'),
                )

    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        data = []
        for game in self.games:
            for data_path in self.root.glob(f"game{game}/*/Label.csv"):
                data_df = pd.read_csv(data_path)
                frame_names, visibilities, xy = data_df['file name'].tolist(), data_df['visibility'].tolist(), [xy for xy in zip(data_df['x-coordinate'].tolist(), data_df['y-coordinate'].tolist())]
                step = 1 if self.is_sequential else self.frame_in
                for i in range(0, len(frame_names) - self.frame_in + 1, step):
                    paths = [str(data_path.parent / frame_name) for frame_name in frame_names[i:i + self.frame_in]]
                    vis = visibilities[i:i + self.frame_in]
                    keypoints = xy[i:i + self.frame_in]
                    data.append((paths, vis, keypoints))
        return data
    
    def __getitem__(self, index):
        paths, visibilities, keypoints = self.data[index]
        imgs = []
        heat_maps = []
        annos = []
        annos_transformed = []
        vises = []
        for path, keypoint, vis in zip(paths, keypoints, visibilities):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if vis:
                transformed = self.transform(image = img, keypoints = [(keypoint[0] - 0.1, keypoint[1] - 0.1)])
                keypoint_transformed = transformed['keypoints']
            else:
                transformed = self.transform(image = img, keypoints = [[0, 0]])
                keypoint_transformed = []
            img = transformed['image']
            if keypoint_transformed:
                heat_map = gen_binary_map((img.shape[2], img.shape[1]), keypoint_transformed[0], self.r)
                vises.append(1)
            else:
                heat_map = np.zeros(shape = (img.shape[1], img.shape[2]))
                keypoint_transformed.append([-1, -1])
                vises.append(0)
            imgs.append(img)
            heat_maps.append(torch.tensor(heat_map))
            annos.append(list(keypoint))
            annos_transformed.append(list(keypoint_transformed[0]))
        imgs = torch.cat(imgs)
        heat_maps = torch.stack(heat_maps)
        annos = torch.tensor(annos)
        annos_transformed = torch.tensor(annos_transformed)
        vises = torch.tensor(vises)
        # print(imgs.dtype, heat_maps.dtype, annos.dtype, annos_transformed.dtype)
        return imgs.float(), heat_maps.float(), annos.float(), annos_transformed.float(), vises.float()
    
def get_data_loaders(root, frame_in, is_sequential, batch_size, transform = None, NUM_WORKERS = os.cpu_count()):
    train_dataset = Tennis(root = root, train = True, transform = transform, frame_in = frame_in, is_sequential = is_sequential)
    test_dataset = Tennis(root = root, train = False, transform = transform, frame_in = frame_in, is_sequential = is_sequential)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = False)
    return train_loader, test_loader
    


if __name__ == "__main__":
    root = "D:\\thang\\20232\\thesis\\Dataset\\Dataset"
    train = True
    # transform = A.Compose([
    #     A.Resize(288, 512, p = 1),
    #     A.RandomBrightnessContrast(p = 0.2),
    #     A.HorizontalFlip(p = 0.5),
    #     A.VerticalFlip(p = 0.5),
    #     A.Rotate(limit = 40, p = 0.9),
    #     A.RandomSizedCrop(height = int(288 * 0.8), width = int(512 * 0.8), p = 0.9),
    #     A.Resize(288, 512, p = 1),
    #     A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
    #     ToTensorV2()
    # ], keypoint_params = A.KeypointParams(format = 'xy', remove_invisible = True, angle_in_degrees = True))
    transform = None
    frame_in = 3
    is_sequential = False
    dataset = Tennis(root, train = train, transform = transform, frame_in = frame_in, is_sequential = is_sequential)
    # dataset[200]
    train_loader, test_loader = get_data_loaders(root = root, transform = transform, frame_in = frame_in, is_sequential = is_sequential, batch_size = 2, NUM_WORKERS = 2)
    # for i, (imgs, heat_maps, annos, annos_transformed) in enumerate(test_loader):
    #     # print(imgs, heat_maps, annos, annos_transformed)
    #     print(imgs.shape, heat_maps.shape, annos.shape, annos_transformed.shape)
    print(next(iter(train_loader))[4])