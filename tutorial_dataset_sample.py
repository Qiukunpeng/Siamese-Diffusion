import cv2
import json
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        root = './data/prompt.json'
        with open(root, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = Image.open(source_filename).convert('L')
        source_array = np.array(source)
        threshold = 127
        binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
        binary_image = Image.fromarray(binary_array)  
        source = binary_image.convert('RGB')

        target = Image.open(target_filename).convert('RGB')

        source = np.array(source).astype(np.uint8)
        target = np.array(target).astype(np.uint8)

        preprocess = self.transform()(image=target, mask=source)
        source, target = preprocess['mask'], preprocess['image']

        ############ Mask-Image Pair ############
        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


    def transform(self, size=384):
        transforms = albumentations.Compose(
                        [
                            albumentations.Resize(height=size, width=size)
                                                            
                        ]
                    )
        return transforms
    