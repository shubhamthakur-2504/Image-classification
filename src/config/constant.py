from torchvision import transforms
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESIZE = (128, 128)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

TRANSFORM = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

BATCH_SIZE = 32

CLASS_NAMES = {}

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_PATH = "artifacts/model.pth"
CLASS_NAMES_PATH = "artifacts/class_names.json"

EPOCHS = 80
LEARNING_RATE = 0.001
INPUT_DIM = 128