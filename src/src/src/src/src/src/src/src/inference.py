import torch
from PIL import Image
from torchvision import transforms, models
import sys

model = models.resnet50()
model.load_state_dict(torch.load("models/resnet50.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return output.argmax(dim=1).item()

if __name__ == "__main__":
    img_path = sys.argv[1]
    print("Predicted class:", predict(img_path))
