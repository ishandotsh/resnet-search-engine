import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet34
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

trfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = 'data'
dataset = datasets.ImageFolder(data_dir, trfms)
dl = DataLoader(dataset, batch_size=32, shuffle=False)

model = resnet34(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
model.avgpool.register_forward_hook(get_features('data'))

stored_ftrs = []

with torch.no_grad():
    for i, (img, label) in enumerate(dl):
        if (i + 1) % 10 == 0:
            print(i+1)
        img = img.to(device)
        output = model(img)
        stored_ftrs.append(features['data'].cpu().numpy())

np.save('stored_ftrs.npy', stored_ftrs, allow_pickle=True)
