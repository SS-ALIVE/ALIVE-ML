import os
import glob
import torch
import random
import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines
from lipreading.preprocess import NormalizeUtterance
import torchaudio.transforms as transforms

class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz', use_boundary=False):
        assert os.path.isfile( label_fp ), \
            f"File path provided for the labels does not exist. Path iput: {label_fp}."
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = False # set True to use annonation directory
        self.use_boundary = use_boundary
        self.label_idx = -3

        self.preprocessing_func = preprocessing_func

        if self.use_boundary or (self.is_var_length and data_partition == "train"):
            assert self._annonation_direc is not None, \
                "Directory path provided for the sequence timestamp (--annonation-direc) should not be empty."
            assert os.path.isdir(self._annonation_direc), \
                f"Directory path provided for the sequence timestamp (--annonation-direc) does not exist. Directory input: {self._annonation_direc}"

        self._data_files = []

        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)


        # -- add examples to self._data_files
        self._get_files_for_partition()

        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path( x )
            self.list[i] = [ x, self._labels.index( label ) ]
            self.instance_ids[i] = self._get_instance_id_from_path( x )

        print(f"Partition {self._data_partition} loaded")

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        #changed / to \\ for windows
        instance_id = x.split('\\')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        # changed / to \\ for windows
        return x.split('\\')[self.label_idx]

    def _get_files_for_partition(self):
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

 
        # get npy/npz/mp4 files
        search_str_npz = os.path.join(dir_fp, '*', self._data_partition, '*.npz')
        search_str_npy = os.path.join(dir_fp, '*', self._data_partition, '*.npy')
        search_str_mp4 = os.path.join(dir_fp, '*', self._data_partition, '*.mp4')
        
        #replace \ with / - for windows
        search_str_npz=search_str_npz.replace('/','\\')
        search_str_npy=search_str_npy.replace('/','\\')
        search_str_mp4=search_str_mp4.replace('/','\\')
        #print(search_str_mp4,search_str_npy,search_str_npz)
        
        self._data_files.extend( glob.glob( search_str_npz ) )
        self._data_files.extend( glob.glob( search_str_npy ) )
        self._data_files.extend( glob.glob( search_str_mp4 ) )
        #print(self._data_files[10])
        #print(glob.glob(os.path.join(self._data_dir,search_str_mp4)))
        # If we are not using the full set of labels, remove examples for labels not used
        #changed / to \\ for windows
        self._data_files = [ f for f in self._data_files if f.split('\\')[self.label_idx] in self._labels ]

    def load_data(self, filename):

        try:
            if filename.endswith('npz'):
                return np.load(filename)['data'] # temporal
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print(f"Error when reading file: {filename}")
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('\\')[self.label_idx:] )  # swap base folder, changed '/' to '\\' for windows
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)  

        utterance_duration = float( info[4].split(' ')[1] )
        half_interval = int( utterance_duration/2.0 * self.fps)  # num frames of utterance / 2
                
        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1)  )   # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint( min( mid_idx+half_interval+1,n_frames ), n_frames  )

        return raw_data[left_idx:right_idx]


    def _get_boundary(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('\\')[self.label_idx:] )  # swap base folder, changed '/' to '\\' for windows
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float( info[4].split(' ')[1] )
        # boundary is used for the features at the top of ResNet, which as a frame rate of 25fps.
        if self.fps == 25:
            half_interval = int( utterance_duration/2.0 * self.fps)
            n_frames = raw_data.shape[0]
        elif self.fps == 16000:
            half_interval = int( utterance_duration/2.0 * 25)
            n_frames = raw_data.shape[0] // 640

        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx-half_interval-1)
        right_idx = min(mid_idx+half_interval+1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary

    def __getitem__(self, idx):

        raw_data = self.load_data(self.list[idx][0])
        # -- perform variable length on training set
        if ( self._data_partition == 'train' ) and self.is_var_length and not self.use_boundary:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        if self.use_boundary:
            boundary = self._get_boundary(self.list[idx][0], raw_data)
            return preprocess_data, label, boundary
        else:
            return preprocess_data, label

    def __len__(self):
        return len(self._data_files)

class AVDataset(object): # dataset for multi-modal training
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz', use_boundary=False):
        assert os.path.isfile( label_fp ), \
            f"File path provided for the labels does not exist. Path iput: {label_fp}."
        self._data_partition = data_partition
        self._data_dir = data_dir ## audio & visual directory..
        self._audio_data_dir = os.path.join(data_dir,'audio_data')
        self._video_data_dir = os.path.join(data_dir,'video_data')
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.video_fps = 25
        self.audio_fps = 16000
        self.is_var_length = False # set True to use annonation directory
        self.use_boundary = use_boundary
        self.label_idx = -3

        self.audio_preprocessing_func = preprocessing_func['audio'][data_partition]
        self.video_preprocessing_func = preprocessing_func['video'][data_partition]

        if self.use_boundary or (self.is_var_length and data_partition == "train"):
            assert self._annonation_direc is not None, \
                "Directory path provided for the sequence timestamp (--annonation-direc) should not be empty."
            assert os.path.isdir(self._annonation_direc), \
                f"Directory path provided for the sequence timestamp (--annonation-direc) does not exist. Directory input: {self._annonation_direc}"

        self._audio_data_files = []
        self._video_data_files = []

        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self._label_fp)


        # -- add examples to self._data_files
        self._get_files_for_partition(True) # audio data file partitioning
        self._get_files_for_partition(False) # video data file partitioning

        # -- from self._data_files to self.list # video,audio.
        self.audio_list = dict()
        self.audio_instance_ids = dict()
        self.video_list = dict()
        self.video_instance_ids = dict()
        for i, x in enumerate(self._audio_data_files): # list an instance_id can be shared, thus call only once in audio loading.
            label = self._get_label_from_path( x )
            self.audio_list[i] = [ x, self._labels.index( label ) ]
            self.audio_instance_ids[i] = self._get_instance_id_from_path( x )
        for i, x in enumerate(self._video_data_files): # list an instance_id can be shared, thus call only once in audio loading.
            label = self._get_label_from_path( x )
            self.video_list[i] = [ x, self._labels.index( label ) ]
            self.video_instance_ids[i] = self._get_instance_id_from_path( x )
        print(f"Partition {self._data_partition} loaded")

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        #changed / to \\ for windows
        instance_id = x.split('\\')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        # changed / to \\ for windows
        return x.split('\\')[self.label_idx]

    def _get_files_for_partition(self,is_audio=False):
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

 
        # get npy/npz/mp4 files
        search_str_npz = os.path.join(dir_fp,"audio_data" if is_audio else "visual_data", '*', self._data_partition, '*.npz')
        search_str_npy = os.path.join(dir_fp,"audio_data" if is_audio else "visual_data", '*', self._data_partition, '*.npy')
        search_str_mp4 = os.path.join(dir_fp,"audio_data" if is_audio else "visual_data", '*', self._data_partition, '*.mp4')

        #replace \ with / - for windows
        search_str_npz=search_str_npz.replace('/','\\')
        search_str_npy=search_str_npy.replace('/','\\')
        search_str_mp4=search_str_mp4.replace('/','\\')
        #print(search_str_mp4,search_str_npy,search_str_npz)
        if is_audio:
            self._audio_data_files.extend( glob.glob( search_str_npz ) )
            self._audio_data_files.extend( glob.glob( search_str_npy ) )
            self._audio_data_files.extend( glob.glob( search_str_mp4 ) )
            self._audio_data_files = [ f for f in self._audio_data_files if f.split('\\')[self.label_idx] in self._labels ]
        else:
            self._video_data_files.extend( glob.glob( search_str_npz ) )
            self._video_data_files.extend( glob.glob( search_str_npy ) )
            self._video_data_files.extend( glob.glob( search_str_mp4 ) )
            self._video_data_files = [ f for f in self._video_data_files if f.split('\\')[self.label_idx] in self._labels ]
        #print(self._data_files[10])
        #print(glob.glob(os.path.join(self._data_dir,search_str_mp4)))
        # If we are not using the full set of labels, remove examples for labels not used
        #changed / to \\ for windows
        

    def load_data(self, filename):

        try:
            if filename.endswith('npz'):
                return np.load(filename)['data'] # temporal
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print(f"Error when reading file: {filename}")
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('\\')[self.label_idx:] )  # swap base folder, changed '/' to '\\' for windows
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)  

        utterance_duration = float( info[4].split(' ')[1] )
        half_interval = int( utterance_duration/2.0 * self.fps)  # num frames of utterance / 2
                
        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1)  )   # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint( min( mid_idx+half_interval+1,n_frames ), n_frames  )

        return raw_data[left_idx:right_idx]


    def _get_boundary(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('\\')[self.label_idx:] )  # swap base folder, changed '/' to '\\' for windows
        info_txt = os.path.splitext( info_txt )[0] + '.txt'   # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float( info[4].split(' ')[1] )
        # boundary is used for the features at the top of ResNet, which as a frame rate of 25fps.
        if self.fps == 25:
            half_interval = int( utterance_duration/2.0 * self.fps)
            n_frames = raw_data.shape[0]
        elif self.fps == 16000:
            half_interval = int( utterance_duration/2.0 * 25)
            n_frames = raw_data.shape[0] // 640

        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx-half_interval-1)
        right_idx = min(mid_idx+half_interval+1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary

    def __getitem__(self, idx):
        audio_raw_data = self.load_data(self.audio_list[idx][0])
        video_raw_data = self.load_data(self.video_list[idx][0])
        # -- perform variable length on training set
        if ( self._data_partition == 'train' ) and self.is_var_length and not self.use_boundary:
            data = self._apply_variable_length_aug(self.audio_list[idx][0], raw_data)
        else:
            audio_data = audio_raw_data
            video_data = video_raw_data
        audio_preprocess_data = self.audio_preprocessing_func(audio_data)
        video_preprocess_data = self.video_preprocessing_func(video_data)
        audio_raw_data = NormalizeUtterance()(audio_raw_data)
        label = self.audio_list[idx][1] # labels are same.
        if self.use_boundary: # we do not consider boundaries in cross-modal
            boundary = self._get_boundary(self.audio_list[idx][0], raw_data)
            return preprocess_data, label, boundary
        else:
            return audio_preprocess_data,video_preprocess_data,audio_raw_data # we don't need label. we need original audio data without mixup!

    def __len__(self):
        return len(self._audio_data_files) # video will be same

def pad_packed_collate(batch):
    if len(batch[0]) == 2:
        use_boundary = False
        data_tuple, lengths, labels_tuple = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    elif len(batch[0]) == 3:
        use_boundary = True
        data_tuple, lengths, labels_tuple, boundaries_tuple = zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

    if data_tuple[0].ndim == 1:
        max_len = data_tuple[0].shape[0]
        data_np = np.zeros((len(data_tuple), max_len))
    elif data_tuple[0].ndim == 3:
        max_len, h, w = data_tuple[0].shape
        data_np = np.zeros((len(data_tuple), max_len, h, w))
    for idx in range( len(data_np)):
        data_np[idx][:data_tuple[idx].shape[0]] = data_tuple[idx]
    data = torch.FloatTensor(data_np)

    if use_boundary:
        boundaries_np = np.zeros((len(boundaries_tuple), len(boundaries_tuple[0])))
        for idx in range(len(data_np)):
            boundaries_np[idx] = boundaries_tuple[idx]
        boundaries = torch.FloatTensor(boundaries_np).unsqueeze(-1)
    labels = torch.LongTensor(labels_tuple)

    if use_boundary:
        return data, lengths, labels, boundaries
    else:
        return data, lengths, labels

def av_pad_packed_collate(batch): ## our av collate function #TODO implement faster tensor transforms
    if len(batch[0]) == 3: # audio_preprocessed,video_preprocessed,audio_raw
        use_boundary = False
        audio_data_tuple, audio_lengths, video_data_tuple, video_lengths,audio_raw_data = zip(*[(a, a.shape[0], b, b.shape[0],c) for (a, b,c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    
    max_len = audio_raw_data[0].shape[0]
    audio_data_np = np.zeros((len(audio_raw_data), max_len))

    for idx in range( len(audio_raw_data)):
        audio_raw_data[idx][:audio_raw_data[idx].shape[0]] = audio_raw_data[idx]

    #audio_raw_mel = np.array(mel_transform(audio_raw_data)) # transform padded audio_raw_data into melspectrogram
    max_len = audio_data_tuple[0].shape[0]
    audio_data_np = np.zeros((len(audio_data_tuple), max_len))

    for idx in range( len(audio_data_np)):
        audio_data_np[idx][:audio_data_tuple[idx].shape[0]] = audio_data_tuple[idx]

    max_len, h, w = video_data_tuple[0].shape
    video_data_np = np.zeros((len(video_data_tuple), max_len, h, w))

    for idx in range( len(video_data_np)):
        video_data_np[idx][:video_data_tuple[idx].shape[0]] = video_data_tuple[idx]


    # Create a PyTorch tensor from the list of NumPy arrays
    # audio_data_list = [torch.from_numpy(arr,dtype=torch.float) for arr in audio_data_np]
    # audio_data = torch.stack(audio_data_list)

    # video_data_list = [torch.from_numpy(arr,dtype=torch.float) for arr in video_data_np]
    # video_data = torch.stack(video_data_list)

    # audio_raw_data_list = [torch.from_numpy(arr,dtype=torch.float) for arr in audio_raw_data]
    # audio_raw_data = torch.stack(audio_raw_data_list)
    audio_data = torch.FloatTensor(np.array(audio_data_np)) ##TODO need to transform ndarray to set off arrays for faster tensor transform
    video_data = torch.FloatTensor(np.array(video_data_np))
    audio_raw_data = torch.FloatTensor(np.array(audio_raw_data))
    #audio_raw_mel = torch.FloatTensor(audio_raw_mel)
    audio_raw_mel = mel_transform(audio_raw_data)


    return audio_data, video_data, audio_lengths, video_lengths,audio_raw_mel#,audio_raw_data # temp 

def mel_transform(batch_data): ## transform audio_raw_data into mel_spectrogram => something's wrong #TODO
    mel_trans = transforms.MelSpectrogram(
                sample_rate = 16000,
                n_fft=1024,
                hop_length=145,
                n_mels=128
            )
    return mel_trans(batch_data)[:,:,:128]