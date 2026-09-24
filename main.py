from fastapi import FastAPI, File, UploadFile
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import io
import base64
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class FashionCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )
    def forward(self, x):
        return self.classifier(self.block_2(self.block_1(x)))


app = FastAPI()
model = FashionCNN(1, 10, 10)
model.load_state_dict(torch.load("fashion_cnn_model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
image_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


cam = GradCAM(model=model, target_layers=[model.block_2[2]])

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L") 
    image = ImageOps.invert(image) 
    
    tensor = image_transform(image).unsqueeze(0)
    

    with torch.inference_mode():
        logits = model(tensor)
        prediction_index = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0][prediction_index].item()
    

    targets = [ClassifierOutputTarget(prediction_index)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    

    img_rgb = tensor.squeeze().numpy()
    img_rgb = np.repeat(img_rgb[:, :, np.newaxis], 3, axis=2)

    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    heatmap_pil = Image.fromarray(visualization)
    

    buffered = io.BytesIO()
    heatmap_pil.save(buffered, format="PNG")

    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "class": classes[prediction_index],
        "confidence": round(confidence * 100, 2),
        "heatmap_base64": img_str
    }