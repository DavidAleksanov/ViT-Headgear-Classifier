import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import ViT


# Specify paths to your data
dataset = {
    'train_data': '/path/to/your/train_data',
    'valid_data': '/path/to/your/valid_data',
    'test_data': '/path/to/your/test_data'
}

def load_data(data_dir):
    data = {'imgpath': [], 'labels': []}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and (file.endswith('.jpg') or file.endswith('.png')):
                    data['imgpath'].append(file_path)
                    data['labels'].append(folder)
    return pd.DataFrame(data)

train_df = load_data(dataset['train_data'])
valid_df = load_data(dataset['valid_data'])
test_df = load_data(dataset['test_data'])

lb = LabelEncoder()
train_df['encoded_labels'] = lb.fit_transform(train_df['labels'])
valid_df['encoded_labels'] = lb.transform(valid_df['labels'])
test_df['encoded_labels'] = lb.transform(test_df['labels'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class HeadgearDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 2]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = HeadgearDataset(train_df, transform=transform)
valid_dataset = HeadgearDataset(valid_df, transform=transform)
test_dataset = HeadgearDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT(
    image_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    qkv_dim=64,
    mlp_hidden_size=3072,
    n_layers=12,
    n_heads=12,
    n_classes=len(lb.classes_)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        print(f"Validation Loss: {val_loss/len(valid_loader)}, Validation Accuracy: {100 * val_correct / val_total}")

# Функция тестирования
def test(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * test_correct / test_total}")

# Запуск тренировки и тестирования
train(model, train_loader, valid_loader, criterion, optimizer, epochs=10)
test(model, test_loader)
