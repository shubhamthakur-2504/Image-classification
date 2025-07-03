from torchvision import transforms
import torch


TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESIZE = (128, 128)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

BATCH_SIZE = 32

CLASS_NAMES = {}

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"