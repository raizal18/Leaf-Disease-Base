import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

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


def extract_alexnet_feature(img):

    feature = model(transform(img)).detach().numpy()
    return feature

# mod = torch.load('models/alexnet.pt')