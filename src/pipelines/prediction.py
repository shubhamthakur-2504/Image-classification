import os
import cv2
import json
import torch
from PIL import Image
from src.model_architecture.cnn_model import CustomCnnModel
from src.config.constant import DEVICE, TRANSFORM, INPUT_DIM, MODEL_PATH, CLASS_NAMES_PATH
class ImageClassifier:
    def __init__(self):
        with open (CLASS_NAMES_PATH, "r") as f:
            self.class_names = json.load(f)
        self.model = CustomCnnModel(input_dim=INPUT_DIM, num_classes=len(self.class_names)).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transforms = TRANSFORM
    
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        label = self.class_names[str(predicted.item())]
        image = cv2.imread(image_path)
        cv2.putText(image, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        labeled_image_path = "artifacts/labeled_image.jpg"
        cv2.imwrite(labeled_image_path, image)
        cwd = os.getcwd()
        output_path = os.path.join(cwd, labeled_image_path)
        return label, output_path
    
