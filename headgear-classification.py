'''Modules'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import classification_report, f1_score , confusion_matrix


class PatchEmbedder(nn.Module):
    def __init__(
        self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2) 
        return x


class PathEmbedder(nn.Module):
  def __init__(
      self,
      patch_size = 16,
      in_channels = 3,
      embed_dim = 64,
  ) -> None:
      super().__init__()
      self.patch_size = patch_size
      self.proj = nn.Conv2d()

class LinearProjection(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedder = PatchEmbedder(patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x)  
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  
        x += self.pos_embeddings  
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self, embed_dim: int = 768, qkv_dim: int = 64, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.qkv_dim = qkv_dim
        self.scale = qkv_dim ** -0.5
        self.dropout = nn.Dropout(dropout_rate)
        self.wq = nn.Linear(embed_dim, qkv_dim)
        self.wk = nn.Linear(embed_dim, qkv_dim)
        self.wv = nn.Linear(embed_dim, qkv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        x = attn_weights @ v
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim

        self.attn_heads = nn.ModuleList([
            ScaledDotProductAttention(embed_dim, qkv_dim, dropout_rate)
            for _ in range(n_heads)
        ])

        self.projection = nn.Sequential(
            nn.Linear(n_heads * qkv_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_outputs = [attn(x) for attn in self.attn_heads]
        attn_outputs = torch.cat(attn_outputs, dim=-1)
        x = self.projection(attn_outputs)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_size: int = 3072,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_size)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_hidden_size, embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(n_heads, embed_dim, qkv_dim, dropout_rate)
        self.mlp = MLP(embed_dim, mlp_hidden_size, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att = self.attention(self.layer_norm1(x) + x)
        x = x + x_att
        x_mlp = self.mlp(self.layer_norm2(x) + x)
        x = x + x_mlp
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.linear_projection = LinearProjection(
            image_size, patch_size, in_channels, embed_dim
        )
        self.encoder = nn.ModuleList([
            EncoderBlock(n_heads, embed_dim, qkv_dim, mlp_hidden_size)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_projection(x)
        for block in self.encoder:
            x = block(x)
        x = x[:, 0]  
        x = self.classifier(x)
        return x


'''read data'''

dataset = {
    'train_data': '/Users/david/Downloads/archive/train',
    'valid_data': '/Users/david/Downloads/archive/valid',
    'test_data': '/Users/david/Downloads/archive/test'
}

all_data = {}

for key, path in dataset.items():
    data = {'imgpath': [], 'labels': []}
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        continue

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and (file.endswith('.jpg') or file.endswith('.png')):
                    data['imgpath'].append(file_path)
                    data['labels'].append(folder)

    all_data[key] = data

train_df = pd.DataFrame(all_data['train_data'])
valid_df = pd.DataFrame(all_data['valid_data'])
test_df = pd.DataFrame(all_data['test_data'])

lb = LabelEncoder()
train_df['encoded_labels'] = lb.fit_transform(train_df['labels'])
valid_df['encoded_labels'] = lb.transform(valid_df['labels'])
test_df['encoded_labels'] = lb.transform(test_df['labels'])

print("----------Train-------------")
print(train_df[["imgpath", "labels"]].head(5))
print(train_df.shape)
print("--------Validation----------")
print(valid_df[["imgpath", "labels"]].head(5))
print(valid_df.shape)
print("----------Test--------------")
print(test_df[["imgpath", "labels"]].head(5))
print(test_df.shape)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


'''model training'''

train_path = '/Users/david/Downloads/archive/train'
valid_path = '/Users/david/Downloads/archive/valid'
test_path = '/Users/david/Downloads/archive/test'

train_df = load_data(train_path)
valid_df = load_data(valid_path)
test_df = load_data(test_path)

lb = LabelEncoder()
train_df['encoded_labels'] = lb.fit_transform(train_df['labels'])
valid_df['encoded_labels'] = lb.transform(valid_df['labels'])
test_df['encoded_labels'] = lb.transform(test_df['labels'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = HeadgearDataset(train_df, transform=transform)
valid_dataset = HeadgearDataset(valid_df, transform=transform)
test_dataset = HeadgearDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

train(model, train_loader, valid_loader, criterion, optimizer, epochs=10)

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

print(test(model, test_loader))