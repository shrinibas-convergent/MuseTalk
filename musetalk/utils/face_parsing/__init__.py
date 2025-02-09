import torch
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet
import torchvision.transforms as transforms

class FaceParsing:
    def __init__(self, 
                 resnet_path='./models/face-parse-bisent/resnet18-5c106cde.pth', 
                 model_pth='./models/face-parse-bisent/79999_iter.pth'):
        # Store device once for all operations.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.model_init(resnet_path, model_pth)
        self.preprocess = self.image_preprocess()

    def model_init(self, resnet_path, model_pth):
        net = BiSeNet(resnet_path)
        # Load model weights mapping to self.device.
        checkpoint = torch.load(model_pth, map_location=self.device)
        net.load_state_dict(checkpoint)
        net.to(self.device)
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512)):
        # Accept a file path or an image array.
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume numpy array is in BGR format; convert to PIL RGB.
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported image type")

        # Resize the image.
        image_resized = image.resize(size, Image.BILINEAR)
        # Preprocess to tensor.
        img_tensor = self.preprocess(image_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.net(img_tensor)[0]
        
        # Squeeze and compute the argmax along the channel dimension.
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # Set all values in [1, 13] to 255, and others to 0.
        parsing = np.where((parsing >= 1) & (parsing <= 13), 255, 0).astype(np.uint8)
        return Image.fromarray(parsing)

if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')
