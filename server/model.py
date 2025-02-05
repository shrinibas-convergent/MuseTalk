# model.py
import torch
from musetalk.utils.utils import load_all_model

# Load the model weights (this happens once at startup)
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

# Switch models to half precision for faster inference
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()
