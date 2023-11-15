import glob
import imageio
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import albumentations as A
import tqdm
import random
import imageio
import argparse

parser = argparse.ArgumentParser(description='Train a MRI model.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

args = parser.parse_args()

# 使用解析到的參數
learning_rate = args.lr
num_epochs = args.epochs

ffs = glob.glob('train_data/*/*')

data = []
for f in ffs:
  if 'Brain_Tumor' in f:
    data.append([f, 1])
  else:
    data.append([f, 0])

print(data[4000])
print(data[1000])
# make the ill to label1, healthy to label0

#Using Resnet to reduce training time
class MRICNN_ResNet(nn.Module):
    def __init__(self):
        super(MRICNN_ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # ResNet is for RGB, so change the first conv to channel 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = MRICNN_ResNet().to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()
epochs = 100 

transform = A.Compose([
    A.Resize(224, 224), # Must be 224x224
    A.ShiftScaleRotate(p=0.5),
    A.OpticalDistortion(p=0.5),
    A.GridDistortion(p=0.5),
])


for epoch in range(num_epochs):
    loss_list = []
    random.shuffle(data)
    for f, label in tqdm.tqdm(data):
        im = imageio.imread(f)

        im = Image.fromarray(im).convert('L')
        im = np.array(im)

        im = transform(image=im)['image']

        im = np.expand_dims(im, axis=0)  # 1, 224, 224
        

        im_d = torch.from_numpy(im.astype(np.float32)/255.0).to(device).unsqueeze(0).float()
        images = im_d.to(device)
        #print("Shape of the image tensor:", images.shape)
        labels = torch.from_numpy(np.array([label])).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {np.mean(loss_list)}')

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'/model/MRI_epoch_{epoch}.pt')
torch.save(model.state_dict(), f'/model/MRI_epoch_{epoch}.pt')