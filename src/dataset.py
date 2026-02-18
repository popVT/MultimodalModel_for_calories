import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class MultimodalDataset(Dataset):
    def __init__(self, df, config, transforms=None):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.config = config
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = os.path.join("data/images", row['dish_id'], "rgb.png")
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
            
        emb = self.tokenizer(
            row['ingredients_list'], 
            truncation=True, 
            max_length=128, 
            padding='max_length', 
            return_tensors='pt'
        )

        calory = torch.tensor(row['total_calories'], dtype=torch.float32)
        mass = torch.tensor(row['total_mass'], dtype=torch.float32)

        return {
            'input_ids': emb['input_ids'].squeeze(),
            'attention_mask': emb['attention_mask'].squeeze(),
            'calory': calory,
            'mass': mass,
            'dish_id': row['dish_id'],
            'image': image
        }


def get_dataloader(config, batch_size=8):

    df = pd.read_csv(config.DISH_CSV)
    train = df[df.split == 'train'].reset_index(drop=True)
    val = df[df.split == 'val'].reset_index(drop=True)
    test = df[df.split == 'test'].reset_index(drop=True)

    dummy_model = timm.create_model(config.IMAGE_MODEL_NAME, pretrained=False)
    config_model = resolve_data_config({}, model=dummy_model)

    train_transform = create_transform(**config_model, is_training=True)
    val_transform = create_transform(**config_model, is_training=False)

    train_dataset = MultimodalDataset(train, config, train_transform)
    val_dataset = MultimodalDataset(val, config, val_transform)
    test_dataset = MultimodalDataset(test, config, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader