import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loaders.dataloaders import get_dataloader
from src.model_architecture.cnn_model import CustomCnnModel
from src.config.constant import DEVICE, BATCH_SIZE, TRANSFORM, TRAIN_DIR, TEST_DIR, LEARNING_RATE, EPOCHS, CLASS_NAMES, INPUT_DIM, MODEL_PATH, CLASS_NAMES_PATH
import json

def train_model():
    device = DEVICE
    train_loader, test_loader = get_dataloader(TEST_DIR, TRAIN_DIR, TRANSFORM, BATCH_SIZE)
    num_classes = len(CLASS_NAMES)
    with open (CLASS_NAMES_PATH, "w") as f:
        json.dump(CLASS_NAMES, f)
    model = CustomCnnModel(INPUT_DIM, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs = EPOCHS
    # training loop

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        print(f"epoch {epoch+1}/{epochs}, loss: {running_loss/len(train_loader)}")

    #to evaluate model
    print("evaluating model on test data please wait...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test accuracy is {test_accuracy:.2f} %")

    # to save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': test_accuracy,
        'epoch': epochs
    }, MODEL_PATH)

    print("model saved successfully")