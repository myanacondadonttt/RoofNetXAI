from functools import partial
import torch
from torchvision import transforms
import sys
sys.path.append("/Users/Lisa/Desktop/Master Thesis/RoofNetXAI/roofnet/models")
from vae import VAE
from PIL import Image
import PIL
# import cv2
import numpy as np

class GradCAMVAE:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def resize_image(self, image):
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
        
        if isinstance(image, PIL.Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")

        if image.ndim == 4:
            image = np.squeeze(image, axis=0)  # Remove the batch dimension

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor()  # Convert the image to a tensor
        ])

        resized_image = transform(image).unsqueeze(0).to(self.device)
        return resized_image


    def get_feature_maps(self, image, target_layer):
        activations = {}
        hooks = []

        def hook_fn(module, input, output, key):
            activations[key] = output.detach()

        for name, module in self.model.named_modules():
            if name == target_layer:
                hook = module.register_forward_hook(partial(hook_fn, key=target_layer))
                hooks.append(hook)

        self.model.eval()
        with torch.no_grad():
            image = self.resize_image(image)
            _ = self.model(image)

        feature_maps = activations[target_layer]
        for hook in hooks:
            hook.remove()

        return feature_maps

    def generate_gradcam(self, input_images, target_layer="encoder"):
        self.model.eval()
        gradcam_maps = []

        for image in input_images:
            resized_image = self.resize_image(image)
            resized_feature_maps = self.get_feature_maps(resized_image, target_layer)
            output = self.model.decode(resized_feature_maps)  # Decode the resized feature maps
            gradcam_map = self.compute_gradcam(resized_feature_maps, output)
            gradcam_maps.append(gradcam_map)

        return gradcam_maps
