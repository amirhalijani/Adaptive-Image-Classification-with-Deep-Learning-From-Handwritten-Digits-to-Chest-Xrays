import torch
from chest_xray import ChestXrayNet
from PIL import Image
from torchvision import transforms

model = ChestXrayNet()
model.load_state_dict(torch.load('chest_xray_weights.pt'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image_path = './chest_xray_data/test/normal/NORMAL2-IM-0364-0001.jpeg'
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)

_, predicted_class = torch.max(output, 1)
print(f'Predicted Class: {predicted_class.item()}')

