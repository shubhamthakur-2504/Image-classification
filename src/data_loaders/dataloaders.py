import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.config.constant import CLASS_NAMES


class ImageDataset(Dataset):
  def __init__(self, image_dir, transforms=None) -> None:
    self.image_dir = image_dir
    self.image_path = []
    self.label = []
    self.class_name = {}
    self.transforms = transforms

    for label , class_dir in enumerate(os.listdir(image_dir)):
      self.class_name[label] = class_dir
      class_path = os.path.join(image_dir, class_dir)
      for image in os.listdir(class_path):
        self.image_path.append(os.path.join(class_path, image))
        self.label.append(label)
    
    CLASS_NAMES.update(self.class_name)

  def __len__(self):
    return len(self.image_path)

  def __getitem__(self, index):
    image_path = self.image_path[index]
    image = Image.open(image_path).convert("RGB")
    label = self.label[index]

    if self.transforms:
      image = self.transforms(image)

    return image, label
  
def get_dataloader(test_dir, train_dir, transform = None, batch_size = 32):
  train_dataset = ImageDataset(train_dir, transform)
  test_dataset = ImageDataset(test_dir, transform)

  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

  return train_loader, test_loader