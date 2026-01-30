import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet50()
model.load_state_dict(torch.load("models/resnet50.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def classify_image(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return f"Predicted Class Index: {output.argmax(dim=1).item()}"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Hybrid Ensemble Vision System",
    description="Upload a product image to classify using CNN + GPT ensemble model."
)

interface.launch()
