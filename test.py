import torch
from model import AVLipreading
from main import get_model_from_json
model_path = "./train_logs/tcn/2023-05-21T17;03;02/ckpt.pth"
av_model = get_model_from_json()
model = torch.load(model_path)
loaded_state_dict = checkpoint['model_state_dict']
model = torch.load_state_dict(loaded_state_dict)