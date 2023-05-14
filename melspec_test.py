import torch,torchaudio
import librosa
import torchaudio.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
audio_path = 'datasets\\audio_data\\ABOUT\\train\\ABOUT_00001.npz'
temp = np.load(audio_path)
temp = torch.tensor(temp['data'])
mel_transform = transforms.MelSpectrogram(
    sample_rate = 16000,
    n_fft=400,
    hop_length=160,
    n_mels=128
)
mel_specs = mel_transform(temp)
print(mel_specs.shape)
plt.figure(figsize=(10, 4))
plt.imshow(torch.log(mel_specs), aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Frames')
plt.ylabel('Mel Filterbanks')
plt.tight_layout()
plt.show()

