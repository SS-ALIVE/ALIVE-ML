import torch,torchaudio
import librosa
import torchaudio.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile
#sample_rate = 16000  

audio_path = 'datasets\\audio_data\\ABOUT\\train\\ABOUT_00001.npz'
temp = np.load(audio_path)
temp = torch.tensor(temp['data'])
#temp = temp['data']
#wavfile.write('output.wav', sample_rate,temp)
print(torch.max(temp),torch.min(temp))
print(temp)
# mel_transform = transforms.MelSpectrogram(
#     sample_rate = 16000,
#     n_fft=400,
#     hop_length=640,
#     n_mels=128
# )
# s1 = time.time()
# mel_specs = mel_transform(temp)
# s2 = time.time() - s1
# print(s2)

# ## visulization
# plt.figure(figsize=(10, 4))
# plt.imshow(torch.log(mel_specs), aspect='auto', origin='lower')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.xlabel('Frames')
# plt.ylabel('Mel Filterbanks')
# plt.tight_layout()
# plt.show()
