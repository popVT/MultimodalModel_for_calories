import torch

class Config:
    SEED = 42
    IMAGE_DIR = "data/images"
    DISH_CSV = "data/dish_cleaned.csv"
    INGREDIENTS_CSV = "data/ingredients.csv"

    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "efficientnet_b3"
    HIDDEN_DIM = 256
    IMAGE_FEATURES_DIM = 1536
    TEXT_MODEL_UNFREEZE = "encoder.layer.11"
    IMAGE_MODEL_UNFREEZE = "blocks.5|conv_head|bn2"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    EPOCHS = 30
    TEXT_LR = 2e-5
    IMAGE_LR = 1e-4
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SAVE_PATH = "best_calorie_model.pth"
    CHECKPOINT_PATH = "calorie_training_checkpoint.pth"
