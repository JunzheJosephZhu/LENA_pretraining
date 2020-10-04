import torch
import os
from encoder_decoder import Encoder, Model
root = "/ws/ifp-10_3/hasegawa/junzhez2/LENA_pretraining"
load_path = "train/checkpoints/latest_model.tar"
output_path = "pretrained_encoder.pth"

pkg = torch.load(os.path.join(root, load_path), map_location='cpu')
print(pkg.keys())
model = Model()
model.load_state_dict(pkg['model'])
encoder = model.encoder
output_pkg = {"state_dict": encoder.state_dict()}
torch.save(output_pkg, os.path.join(root, output_path))