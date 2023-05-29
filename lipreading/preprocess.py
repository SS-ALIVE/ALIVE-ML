import cv2
import random
import numpy as np
import os

__all__ = ['Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'RandomCrop',
           'HorizontalFlip', 'AddNoise', 'NormalizeUtterance', 'TimeMask']


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class NormalizeUtterance():
    """Normalize per raw audio by removing the mean and divided by the standard deviation
    """
    def __call__(self, signal):
        signal_std = 0. if np.std(signal)==0. else np.std(signal)
        signal_mean = np.mean(signal)
        #return (signal - signal_mean) / signal_std
        return signal ## temporal code - not using normalization!


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """
    
    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20,9999]): 
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal


class AddAudioNoise(object):
    """Add another train audio as noise"""
    # train 시에만 사용
    def __init__(self):
        pass

    def __call__(self, signal):
        signal = signal['data'] # .npz to numpy array

        # find a random noise file
        folder_path = '..\\datasets\\audio_data\\'
        file_list = os.listdir(folder_path) 
        random_folder = random.choice(file_list)
        file_list = os.listdir(folder_path+random_folder+"\\train\\")
        random_file = random.choice(file_list)
        noise_data = np.load(random_file)['data']

        src_len = len(signal)
        noise_len = len(noise_data)

        start_time = random.randint(-noise_len, src_len) # 겹치는 위치

        if start_time < 0:
            noise_data = noise_data[-start_time:]
            noise_data = np.pad(noise_data, (0, src_len-start_time-noise_len), mode='constant', constant_values = 0)
        else:
            noise_data = noise_data[:src_len - start_time]
            noise_data = np.pad(noise_data, (start_time, 0), mode='constant', constant_values = 0)
        
        # noise의 크기 : 0~0.5 랜덤 값
        noised_signal = signal + random.random()*0.5*noise_data
        return noised_signal
    

class TimeMask():
    """time mask
    """
    def __init__(self, T=6400, n_mask=2, replace_with_zero=False, inplace=False):
        self.n_mask = n_mask
        self.T = T

        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def __call__(self, x):
        if self.inplace:
            cloned = x
        else:
            cloned = x.copy()

        len_raw = cloned.shape[0]
        ts = np.random.randint(0, self.T, size=(self.n_mask, 2))
        for t, mask_end in ts:
            if len_raw - t <= 0:
                continue
            t_zero = random.randrange(0, len_raw - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            if self.replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned
