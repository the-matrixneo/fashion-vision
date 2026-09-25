# FashionVision: CNN Classification & Explainability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yS-nlTJ_AVkpe9walgLDhRH-_uUOaKAR?usp=sharing)

This is a Computer Vision project I built to classify clothing items using PyTorch. Instead of just stopping at accuracy, I wanted to understand *why* the model makes its decisions, so I used Grad-CAM to visualize exactly what the neural network is looking at. Finally, I wrapped the model in a FastAPI backend so it can process real-world images.

## The Model (TinyVGG)
Built a custom Convolutional Neural Network (CNN) because it preserves the 2D shapes of the clothing, rather than flattening the image immediately. 

* **Convolutions:** Two layers of `nn.Conv2d` to pick up on edges and shapes (like sleeves or collars).
* **Downsampling:** `nn.MaxPool2d` to shrink the image data down (from 28x28 to 7x7) so the model runs faster and focuses on the most important features.
* **Output:** A final linear layer that outputs predictions for 10 different clothing categories.

## Results & Visualizing the Model
The CNN hit **87.44% accuracy** on the test data in just 3 epochs.

### What it got right (Grad-CAM)
I used Grad-CAM to generate a heatmap over the images. It was great to see that the model actually learned to look at the physical structure of the clothes (like the shoulders) instead of getting distracted by big logos printed on the front.

![Pullover Grad-CAM](assets/pullover_success.png)

### Where it struggled (Confusion Matrix)
The model was great at identifying shoes, but it had a hard time telling the difference between upper-body clothes like Coats, Shirts, and Pullovers since their structural shapes are so similar.

![Confusion Matrix](assets/confusion_matrix.png)

### Fixing Real-World Mistakes
When I tried passing a picture of a shoe I downloaded from Google, the model completely failed and guessed "Bag". 
        
Using Grad-CAM, I realized the model was confused by the white background of the web image (the training data used black backgrounds). To fix this, I updated the API to automatically invert the colors (`1.0 - x`) of incoming images to match the training data. After that, it worked perfectly!

![Grad-CAM Distribution Shift](assets/grad_cam.png)

## The API (FastAPI)
To make the model usable, I served the trained PyTorch weights using FastAPI.

* Automatically resizes and color-inverts incoming images so they match the training data.
* Runs the CNN prediction.
* Generates the Grad-CAM heatmap and sends it back as a Base64 image so a frontend can easily display it.

### How to run it locally
```bash
# Install the required packages
pip install fastapi uvicorn torch torchvision pillow grad-cam numpy matplotlib python-multipart

# Start the server
uvicorn main:app --reload