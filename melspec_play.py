import numpy as np
import librosa
import torch
import os
import soundfile as sf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Assuming you have a mel spectrogram image
mel_spectrogram = torch.load('masked_audio.pt').detach().cpu().numpy()  # Replace with your mel spectrogram image data

# Convert mel spectrogram to linear spectrogram
waveform = librosa.feature.inverse.mel_to_audio(mel_spectrogram,sr=16000,n_fft=1024,hop_length =145)


# Apply inverse STFT to obtain the waveform
# inverse_stft = librosa.griffinlim
# waveform = inverse_stft(stft_spec)

# Save the waveform as a WAV file
sf.write('test.wav', waveform, 16000)