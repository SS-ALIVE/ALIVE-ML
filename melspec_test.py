import torch,torchaudio
import librosa
import torchaudio.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
audio_path = 'datasets\\audio_data\\ABOUT\\train\\ABOUT_00001.npz'
temp = np.load(audio_path)
temp = torch.tensor(temp['data'])
mel_transform = transforms.MelSpectrogram(
    sample_rate = 16000,
    n_fft=400,
    hop_length=640,
    n_mels=128
)
s1 = time.time()
mel_specs = mel_transform(temp)
s2 = time.time() - s1
print(s2)

