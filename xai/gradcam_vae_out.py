import torchcam
import sys
sys.path.append("/Users/Lisa/Desktop/Master Thesis/RoofNetXAI/roofnet/models")
from vae import VAE
from torchcam.methods import GradCAM



# Define the GradCAMVAE class that extends the VAE model
class GradCAMVAE(VAE):
    def __init__(self, *args, **kwargs):
        super(GradCAMVAE, self).__init__(*args, **kwargs)
        self.gradcam = GradCAM(self, target_layer='encoder')  # Adjust target_layer if needed

    def generate_gradcam(self, images):
        self.eval()
        gradcam_maps = self.gradcam(images)
        return gradcam_maps