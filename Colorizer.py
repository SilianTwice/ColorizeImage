import os
from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# Настройки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-4
ENSEMBLE_SIZE = 3

DATASET_PATH = r"G:\Muiv\Practika\dataset"
MODELS_DIR = "MODELS"
BLACKWHITE_IMAGE = r"G:\Muiv\Practika\Black and White Dataset\17.jpg"
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

os.makedirs(MODELS_DIR, exist_ok=True)

class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.paths.extend(glob(os.path.join(folder, "**", ext), recursive=True))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        L = lab[:, :, 0:1] / 255.0
        ab = (lab[:, :, 1:] - 128.0) / 128.0

        return (
            torch.tensor(L).permute(2, 0, 1).float(),
            torch.tensor(ab).permute(2, 0, 1).float()
        )

class Model_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU()
        )
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        
        self.out = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        b = self.bottleneck(p4)
        
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

class Model_ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Model_MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        self.encoder = mobilenet.features

        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train():
    
    dataset = ColorizationDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Загружено изображений {len(dataset)}")
    print(f"Устройство {DEVICE}")
    print("- " * 100)

    model_classes = [Model_UNet, Model_ResNet34, Model_MobileNet]
    
    for idx in range(ENSEMBLE_SIZE):
        print(f"Обучение: {model_classes[idx].__name__}")
        print("- " * 100)
        
        model = model_classes[idx]().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            
            for L, ab in tqdm(loader, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
                L, ab = L.to(DEVICE), ab.to(DEVICE)
                
                pred = model(L)
                loss = criterion(pred, ab)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step()
            
            print(f"Потеря {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 
                          os.path.join(MODELS_DIR, f"model_{idx}.pth"))
    
    print("=" * 100)

def ImageColorized(path):
    
    model_classes = [Model_UNet, Model_ResNet34, Model_MobileNet]
    ensemble = []
    
    for idx in range(ENSEMBLE_SIZE):
        model = model_classes[idx]().to(DEVICE)
        model.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, f"model_{idx}.pth"), 
                      map_location=DEVICE, weights_only=True)
        )
        model.eval()
        ensemble.append(model)
    
    
    img = cv2.imread(path)
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2Lab)
    
    L = lab[:, :, 0:1] / 255.0
    L = torch.tensor(L).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    predictions = []
    
    with torch.no_grad():
        for model in ensemble:
            pred = model(L)
            predictions.append(pred)
    
    ab = torch.mean(torch.stack(predictions), dim=0)
    ab = ab[0].permute(1, 2, 0).cpu().numpy()
    
    ab = ab * 140 + 128
    ab = np.clip(ab, 0, 255)
    
    lab_out = np.concatenate([lab[:, :, 0:1], ab], axis=2).astype(np.uint8)
    result = cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB)
    
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(121)
    plt.imshow(cv2.resize(gray, (w, h)), cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=" * 100)
    
    choice = input("Обучить/переобучить? (y/yes/д/да/l/н): ").lower()
    
    if choice in ['y', 'д', 'yes', 'да', 'l', 'н']:
        train()
    else:
        print("Пропуск обучения/переобучения")
    
    print("=" * 100)
    ImageColorized(BLACKWHITE_IMAGE)