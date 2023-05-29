# # import torch
# # from model import AVLipreading
# # from main import get_model_from_json
# # model_path = "./train_logs/tcn/2023-05-21T17;03;02/ckpt.pth"
# # av_model = get_model_from_json()
# # model = torch.load(model_path)
# # loaded_state_dict = checkpoint['model_state_dict']
# # model = torch.load_state_dict(loaded_state_dict)
# from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
# import torch,torchaudio
# g = torch.manual_seed(1)
# preds,_ = torchaudio.load('./reconstructed_waveform.wav')
# target,_ = torchaudio.load('./original_waveform.wav')
# original,_ = torchaudio.load('./noised_waveform.wav')
# stoi = ShortTimeObjectiveIntelligibility(16000, False)
# print(stoi(preds,target)) # output - original
# print(stoi(original,target)) # noise - original)
# print(stoi(preds,original)) # output - noise
print(488766%200)