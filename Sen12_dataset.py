import torch
import os
import imageio.v2 as imageio
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from dataset_utils import random_crop

class Sen12_dataset(Dataset):
    def __init__(self, image_size_1, image_size_2, dataset_path="data/sen1-2", random_seed = 42, data_range = range(0, 282384)):
        self.dataset_path = dataset_path
        self.image_size_1 = image_size_1
        self.image_size_2 = image_size_2
        self.image_pairs_info_list = []

        self.random_seed = random_seed
        self.data_range = data_range

        self.init_img_pair_infos()
        assert len(self.image_pairs_info_list) > 0, f"No image pairs found in {self.dataset_path}"
        
    
    def init_img_pair_infos(self):
        image_pairs_info_list = []
        assert os.path.exists(self.dataset_path), f"Dataset path {self.dataset_path} does not exist"
        dataset_folders = ["ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter"]
        for folder_path in dataset_folders:
            dataset_folder_path = f'{self.dataset_path}/{folder_path}'
            if os.path.exists(dataset_folder_path):
                for i in range(0, 160):
                    sar_folder_path = f'{dataset_folder_path}/s1_{i}'
                    opt_folder_path = f'{dataset_folder_path}/s2_{i}'
                    if os.path.exists(sar_folder_path) and os.path.exists(opt_folder_path):
                        for sar_img_path in os.listdir(sar_folder_path):
                            opt_img_path = sar_img_path.replace('_s1_', '_s2_')
                            sar_img_path = f'{sar_folder_path}/{sar_img_path}'
                            opt_img_path = f'{opt_folder_path}/{opt_img_path}'
                            if os.path.exists(sar_img_path) and os.path.exists(opt_img_path):
                                img_pair_info = {
                                    'img_opt_path': opt_img_path,
                                    'img_sar_path': sar_img_path
                                }
                                image_pairs_info_list.append(img_pair_info)

        data_range_begin = self.data_range.start
        data_range_end = self.data_range.stop
        total_data_count = len(image_pairs_info_list)

        assert data_range_begin in range(0, total_data_count+1) and data_range_end in range(0, total_data_count+1), f"Invalid data range! Total image pairs: {total_data_count}."

        rand = np.random.RandomState(self.random_seed)
        rand.shuffle(image_pairs_info_list)

        self.image_pairs_info_list = image_pairs_info_list[data_range_begin : data_range_end]

    def __len__(self):
        return len(self.image_pairs_info_list)

    def __getitem__(self, idx):
        img_pair_info = self.image_pairs_info_list[idx]

        img_opt_path = img_pair_info['img_opt_path']
        img_sar_path = img_pair_info['img_sar_path']

        # Read images in RGB mode
        img_opt = imageio.imread(img_opt_path, pilmode='RGB')
        img_sar = imageio.imread(img_sar_path, pilmode='RGB')

        #Convert to PIL image
        img_opt = Image.fromarray(img_opt)
        img_sar = Image.fromarray(img_sar)

        # Perform a random crop on img_sar, which also returns the crop's start position (offset)
        img_sar, ground_truth = random_crop(img_sar, crop_size=self.image_size_2)
        
        # Convert the grayscale PIL Images to numpy arrays.
        img_opt = np.transpose(np.array(img_opt), (2, 0, 1)).copy()
        img_sar = np.transpose(np.array(img_sar), (2, 0, 1)).copy()

        # Convert numpy arrays to torch tensors
        img_opt = torch.from_numpy(img_opt).float()
        img_sar = torch.from_numpy(img_sar).float()
        ground_truth = torch.IntTensor(ground_truth)
        return {
            'img_1': img_opt,
            'img_2': img_sar,
            'gt': ground_truth
        }



def test():
    train_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(0, 280384))
    test_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(280384, 282384))

    print(len(train_dataset.image_pairs_info_list))
    print(len(test_dataset.image_pairs_info_list))

if __name__ == "__main__":
    test()
