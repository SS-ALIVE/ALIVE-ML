import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, pad_packed_collate, av_pad_packed_collate, AVDataset
from torch.utils.data import Subset

def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std),
                                    TimeMask(T=0.6*25, n_mask=1)
                                    ])

        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'audio':

        preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('./data/babbleNoise_resample_16K.npy')),
                                    NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    elif modality == "av": #multimodal dataloader preprocessing code
        video_preprocessing = {}
        audio_preprocessing = {}
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        video_preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std),
                                    TimeMask(T=0.6*25, n_mask=1)
                                    ])

        video_preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        video_preprocessing['test'] = video_preprocessing['val']
        

        audio_preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('./data/babbleNoise_resample_16K.npy')),
                                    NormalizeUtterance()])

        audio_preprocessing['val'] = NormalizeUtterance()

        audio_preprocessing['test'] = NormalizeUtterance()


        preprocessing['audio'] = audio_preprocessing
        preprocessing['video'] = video_preprocessing
    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines(args.modality)

    if args.modality == "av":
        partitions = ['test'] if args.test else ['train','val','test']
        dsets = {partition:AVDataset(
            modality = args.modality,
            data_partition = partition,
            data_dir = args.data_dir,
            label_fp = args.label_path,
            annonation_direc = args.annonation_direc,
            preprocessing_func = preprocessing,
            data_suffix = '.npz',
            use_boundary=args.use_boundary,
        ) for partition in partitions}
        dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=av_pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partitions}
        return dset_loaders
    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz',
                use_boundary=args.use_boundary,
                ) for partition in partitions}
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders


def unit_test_data_loader(args): ## unit test dataloader for multimodal(av) loading.
    preprocessing = get_preprocessing_pipelines(args.modality)
    subset_indices = range(256)  # Specify the indices for the desired subset
    if args.modality == "av":
        partitions = ['test'] if args.test else ['train','val','test']
        dsets = {partition:Subset(AVDataset(
            modality = args.modality,
            data_partition = partition,
            data_dir = args.data_dir,
            label_fp = args.label_path,
            annonation_direc = args.annonation_direc,
            preprocessing_func = preprocessing,
            data_suffix = '.npz',
            use_boundary=args.use_boundary,
        ),subset_indices) for partition in partitions}
        dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=av_pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders