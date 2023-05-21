import torch,torchaudio
import librosa
import torchaudio.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.io import wavfile
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sample_rate = 16000  
audio_path = 'datasets\\audio_data\\ABOUT\\train\\ABOUT_00001.npz'

## original audio from npz
temp = np.load(audio_path)
temp = temp['data']
wavfile.write('meltest_original_audio.wav', sample_rate,temp)

## audio to mel reconstruction : using librosa
waveform,sample_rate = librosa.load('./meltest_original_audio.wav')
mel_spectrogram = librosa.feature.melspectrogram(y=waveform,sr=sample_rate) ## mel spec from audio->me
waveform = librosa.feature.inverse.mel_to_audio(mel_spectrogram)
wavfile.write('meltest_reconstructed_audio.wav', sample_rate,waveform)

# audio to mel reconstruction : using torchaudio
# mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate)
# inverse_mel_transform = transforms.InverseMelScale()
# grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=2048)

# mel_spectrogram = mel_transform(torch.tensor(temp)) #audio->mel
# spectrogram = inverse_mel_transform(mel_spectrogram) #mel->spec
# waveform = grifflim_transform(spectrogram) # spec -> wav
# wavfile.write('meltest_reconstructed_audio_torch.wav',sample_rate,waveform.numpy())
# waveform = librosa.feature.inverse.mel_to_audio(melspec.detach().cpu().numpy())
# wavfile.write('meltest_reconstructed_audio_torch.wav',sample_rate,waveform)