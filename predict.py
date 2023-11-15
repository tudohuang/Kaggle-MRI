import glob
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
import pandas as pd
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='Predict MRI images using a trained model.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--input_folder', type=str, required=True, help='Folder where input images are stored')
parser.add_argument('--output_csv', type=str, default='predictions.csv', help='Path to save the predictions CSV')

args = parser.parse_args()

# Load the model
class MRICNN_ResNet(nn.Module):
    def __init__(self):
        super(MRICNN_ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRICNN_ResNet().to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Define the image transformation
transform = A.Compose([
    A.Resize(224, 224), # Images must be resized to 224x224 to match model input
])

# Predict function
def predict(image_path, model, transform):
    image = imageio.imread(image_path)
    image = Image.fromarray(image).convert('L')
    image = np.array(image)
    image = transform(image=image)['image']
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image_tensor = torch.from_numpy(image.astype(np.float32)/255.0).to(device).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    return prediction

# Loop over all images and make predictions
image_files = glob.glob(f'{args.input_folder}/*')
predictions = []

for image_file in image_files:
    prediction = predict(image_file, model, transform)
    image_id = image_file.split('/')[-1]  # or however you want to extract the ID
    predictions.append([image_id, prediction])

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions, columns=['id', 'label'])
predictions_df.to_csv(args.output_csv, index=False)

print(f"Predictions saved to {args.output_csv}")
