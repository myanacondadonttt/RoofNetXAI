import torch
from torchvision import transforms
import sys
sys.path.append("/Users/Lisa/Desktop/Master Thesis/RoofNetXAI/roofnet/models")
from vae import VAE
 
import torch
from torchvision import transforms
# import cv2
import numpy as np

class GradCAMVAE:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    def generate_gradcam(self, input_images, target_layer):  # input_images = batch of images

        # Forward pass through the encoder to obtain the feature maps
        with torch.enable_grad():
            z, mu, logvar, h = self.model.encode(input_images)

        # Convert the feature maps to numpy and resize them to the input image size
        feature_maps = h.detach().cpu().numpy()
        feature_maps = np.expand_dims(feature_maps, axis=1)  # Add channel dimension
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_images.shape[-2], input_images.shape[-1])),
            transforms.ToTensor()
        ])
        resized_feature_maps = torch.stack([transform(fm) for fm in feature_maps])

        # Compute gradients of the output with respect to the feature maps
        resized_feature_maps.requires_grad_()
        output = self.model.decode(resized_feature_maps.to(self.device))
        torch.autograd.backward(output, grad_tensors=torch.ones_like(output))

        # Get the gradients from the resized feature maps
        gradients = resized_feature_maps.grad.clone().detach()

        # Compute the importance weights as the average of the gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Compute the GradCAM maps by element-wise multiplication
        gradcam_maps = torch.mul(resized_feature_maps, weights)

        # Normalize the GradCAM maps
        gradcam_maps = torch.nn.functional.normalize(gradcam_maps, p=2, dim=1)

        return gradcam_maps
