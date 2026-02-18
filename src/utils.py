import re
import os
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel, AutoTokenizer

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import MeanAbsoluteError
from dataclasses import asdict
from src.config import Config
from src.dataset import get_dataloader


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        img_dim = getattr(self.image_model, "num_features", config.IMAGE_FEATURES_DIM)
        self.image_proj = nn.Sequential(
            nn.Linear(img_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.regressor = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_model(input_ids, attention_mask)
        text_features = text_out.last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        img_emb = self.image_proj(image_features)

        combined = torch.cat([text_emb, img_emb], dim=1)
        fused = self.fusion(combined)
        
        calories = self.regressor(fused).squeeze(-1)
        
        return calories


def validate(model, val_loader, device, mae):
    model.eval()
    mae.reset()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            y = batch['calory'].to(device)

            pred = model(input_ids, attention_mask, image)
            mae.update(pred, y)
    
    return mae.compute().item()


def _build_optimizer(config, model):
    # Берём ТОЛЬКО trainable параметры
    text_params = [p for p in model.text_model.parameters() if p.requires_grad]
    img_params = [p for p in model.image_model.parameters() if p.requires_grad]
    head_params = (
        list(model.text_proj.parameters())
        + list(model.image_proj.parameters())
        + list(model.fusion.parameters())
        + list(model.regressor.parameters())
    )

    optimizer = optim.AdamW(
        [
            {"params": text_params, "lr": config.TEXT_LR, "weight_decay": config.WEIGHT_DECAY},
            {"params": img_params, "lr": config.IMAGE_LR, "weight_decay": config.WEIGHT_DECAY},
            {"params": head_params, "lr": config.HEAD_LR, "weight_decay": config.WEIGHT_DECAY},
        ]
    )
    return optimizer


def train(config):
    torch.manual_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloader(config, config.BATCH_SIZE)

    model = MultimodalModel(config).to(device)

    # разморозка слоев
    set_requires_grad(model.text_model, config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, config.IMAGE_MODEL_UNFREEZE)

    optimizer = _build_optimizer(config, model)

    criterion = nn.SmoothL1Loss(beta=20.0)
    #criterion = nn.MSELoss()
    mae = MeanAbsoluteError().to(device)

    best_mae = float("inf")
    train_losses = []
    val_maes = []
    start_epoch = 0

    for epoch in range(start_epoch, config.EPOCHS+start_epoch):
        # обучение
        model.train()
        total_loss = 0.0
        mae.reset()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            y = batch['calory'].to(device)

            optimizer.zero_grad()
            pred = model(input_ids, attention_mask, image)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mae.update(pred, y)

        
        avg_loss = total_loss / len(train_loader)
        train_mae = mae.compute().item()
        train_losses.append(avg_loss)

        # валидация
        val_mae = validate(model, val_loader, device, mae)
        val_maes.append(val_mae)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_Loss: {avg_loss:.4f} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae :.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Новая лучшая модель! MAE = {best_mae:.2f}")

        # Сохранение чекпоинта
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_mae': best_mae,
            'train_losses': train_losses,
            'val_maes': val_maes
        }, config.CHECKPOINT_PATH)
        print(f"Чекпоинт сохранён\n")

    print(f"Обучение завершено. Лучший Val MAE: {best_mae:.2f}")
    
    return model, train_losses, val_maes, best_mae

