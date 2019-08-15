import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelBinarizer
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
import pathlib


class RotorDataset(Dataset):
    
    def __init__(self, data_dir, label_generator):
        """
        Initializes a RotorDataset object, which can be used for tensor manipulations
        :param data_dir:
        """
    
        self.data_dir = data_dir
        self.label_generator = label_generator
        self.label_names = self.label_generator.get_label_names()

        # Initialize label binarizer
        self.label_binarizer = LabelBinarizer()
        label_index_list = list(range(len(self.label_names)))
        self.label_binarizer.fit(label_index_list)
        
        # Initialize data lists
        self.label_list = []
        self.images = []
        
        # Construct data lists
        data_root = pathlib.Path(data_dir)
        all_json_paths = list(data_root.glob('record_*.json'))
        all_json_paths = [str(path) for path in all_json_paths]
        
        for json_path in all_json_paths:
            with open(json_path) as json_file:
                json_data = json.load(json_file)

                # Process image and append to list
                image_filepath = os.path.join(data_dir, json_data['cam/image_array'])
                img = Image.open(image_filepath)
                img = img.convert('RGB')
                img = transforms.functional.resize(img, (64, 64))
                img = transforms.functional.to_tensor(img)
                self.images.append(img)
                
                label_val = label_generator.value_to_label_index(json_data['user/angle'])
                self.label_list.append(label_val)
            
        self.labels = self.label_binarizer.transform(self.label_list).astype(np.float32)
        
    def __getitem__(self, index):
        
        img = self.images[index]
        label = torch.from_numpy(self.labels[index])
        
        return img, label
    
    def __len__(self):
        
        return len(self.images)