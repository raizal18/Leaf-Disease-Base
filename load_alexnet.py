import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


model = models.alexnet(pretrained=True)

model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder('path/to/images', transform=transform)
loader = DataLoader(dataset, batch_size=32)


features = []
with torch.no_grad():
    for images, _ in loader:
        outputs = model(images)
        features.append(outputs)

# Concatenate the extracted features into a single tensor
features = torch.cat(features, dim=0)

# Save the extracted features to a file
torch.save(features, 'features.pt')
mod = torch.load('models/alexnet.pt')