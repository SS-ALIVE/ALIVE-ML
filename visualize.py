import cv2
import dlib
import numpy as np
import librosa
import soundfile as sf
from collections import deque
import subprocess
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torchaudio
import json
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm
import math
import numpy as np
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.resnet1D import ResNet1D, BasicBlock1D
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from lipreading.models.densetcn import DenseTemporalConvNet
from lipreading.models.swish import Swish
from lipreading.models.FCN import FCN
from lipreading.models.ESPCN import ESPCN
from lipreading.models.UNet import UNet
import torchaudio.transforms as transforms
from lipreading.dataset import audio_to_stft
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines, unit_test_data_loader
from visdom import Visdom
import numpy as np
import math
import os.path
import numpy as np
import umap
import torch
import matplotlib.pyplot as plt
 
vis = Visdom()


def _transposed_average_batch(x,lengths,B):
    return torch.stack([torch.mean(x[index][0:i,:],0) for index,i in enumerate(lengths)],0)


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Create a matrix of shape (max_seq_len, d_model) to store the positional encodings
        self.positional_encodings = self._generate_positional_encodings()

    def _generate_positional_encodings(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        # Add positional encodings to the input tensor
        x = x + self.positional_encodings[:, :x.size(1), :].to(x.device)
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention,self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.cross_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads = self.num_heads, batch_first=True) # cross-modality feature
        self.attention_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.feedforward_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
            Swish(),
            nn.Linear(2048, self.embed_dim)
        )
        self.feedforward_layer_norm = nn.LayerNorm(self.embed_dim)


    def forward(self, Q, K, V):
        cross_attention = self.attention_layer_norm(self.cross_attention(Q, K, V)[0])
        # cross_attention = self.cross_attention(Q, K, V)[0] + Q

        
        cross_feedforward = self.feedforward_layer_norm(self.feedforward_layer(cross_attention) + cross_attention)
        # cross_feedforward = self.feedforward_layer(cross_attention) + cross_attention



        # return av_attention, va_attention, a_self_attention,v_self_attention
        return cross_feedforward

class Seperator_Block(nn.Module):

    def __init__(self, num_layers, d_model, n_head):
        super(Seperator_Block, self).__init__()

        #@# transfomer
        self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.video_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer,num_layers=  num_layers)
        self.video_encoder = nn.TransformerEncoder(self.video_encoder_layer,num_layers = num_layers)

        self.av_cross_attention = CrossAttention(embed_dim = d_model, num_heads = n_head)
        self.va_cross_attention = CrossAttention(embed_dim = d_model, num_heads = n_head)
        self.audio_reduction = nn.Sequential( # 1024 512 1 56 29 1024
            nn.Conv1d(d_model*2,d_model,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm1d(d_model),
            Swish()
        )
        self.video_reduction = nn.Sequential(
            nn.Conv1d(d_model*2,d_model,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm1d(d_model),
            Swish()
        )
        self.positional_encodings = PositionalEncoding(d_model = d_model, max_seq_len = 30)
    def forward(self, x):
        audio,video = x
        audio = self.positional_encodings(audio)
        video = self.positional_encodings(video)
        encoded_audio = self.audio_encoder(audio)
        encoded_video = self.video_encoder(video)
        
        av_attention = self.av_cross_attention(encoded_video,encoded_audio,encoded_audio) ## cross-attention , q,k,v -> value
        va_attention = self.va_cross_attention(av_attention,encoded_video,encoded_video)
        
        audio_out = torch.cat((encoded_audio,av_attention),dim=2) #b len feature -> b len feature 2 b len feature * 2
        video_out = torch.cat((encoded_video,va_attention),dim=2) #56 29 1024 56 1024 29

        audio_out = audio_out.transpose(1,2)
        video_out = video_out.transpose(1,2)
        audio_out = self.audio_reduction(audio_out)
        video_out = self.video_reduction(video_out)
        audio_out = audio_out.transpose(1,2)
        video_out = video_out.transpose(1,2)
        # print(audio_out)
        # print(video_out)
        # print(audio_out.shape,video_out.shape)
        # exit()
        audio_out = audio_out + audio # residual
        video_out = video_out + video # residual

        return audio_out,video_out

class AVSep(nn.Module):

    def __init__(self,seperator,d_model,n_head,blocks): # blocks = number of seperator blocks / blocks = [2,3]
        super(AVSep,self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.blocks = blocks
        self.seperator = seperator
        self.layers = self._make_layer(self.seperator,self.d_model,self.blocks,self.n_head)
        
    # seperator[num_layer=2]  -> seperator[num_layer=4]*2  
    def _make_layer(self, transformer_block,d_model,blocks,n_head):
        layers = []
        for num_layers in blocks:
            layers.append(transformer_block(num_layers,d_model,n_head))
        return nn.Sequential(*layers)
    
    def forward(self,audio,video):
        audio,video = self.layers((audio,video)) # audio = (b,18560), video = (b,29,88,88) -> backbone(resnet) -> audio = (b,29,512) / video = (b,29,512)
        
        return audio,video


class AVLipreading_sep_unet(nn.Module):
    def __init__( self, modality='av', hidden_dim=256, backbone_type='resnet', num_classes=500,
                  relu_type='prelu', tcn_options={}, densetcn_options={}, attention_options = {},seperator_options={}, width_mult=1.0,
                  use_boundary=False, extract_feats=False):
        super(AVLipreading_sep_unet, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary

        #multi-modal
        if self.modality == 'av':
            self.frontend_nout = 1
            self.backend_out = 512
            self.audio_trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type) ## feature extraction with npz- > resnet?(audio). is it "best?"
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
                self.video_trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            elif self.backbone_type == 'shufflenet':
                assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
                shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
                self.video_trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
                self.frontend_nout = 24
                self.backend_out = 1024 if width_mult != 2.0 else 2048
                self.stage_out_channels = shufflenet.stage_out_channels[-1]

            # -- frontend3D
            if relu_type == 'relu':
                frontend_relu = nn.ReLU(True)
            elif relu_type == 'prelu':
                frontend_relu = nn.PReLU( self.frontend_nout )
            elif relu_type == 'swish':
                frontend_relu = Swish()

            self.frontend3D = nn.Sequential(
                        nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                        nn.BatchNorm3d(self.frontend_nout),
                        frontend_relu,
                        nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        else:
            raise NotImplementedError

        self.seperator_block = Seperator_Block
        self.seperator = AVSep(
            seperator = self.seperator_block,
            d_model = seperator_options['d_model'],
            n_head =  seperator_options['n_head'],
            blocks = seperator_options['num_layers']
            )
        
        self.consensus_func = _transposed_average_batch

        self.spec_transform = audio_to_stft

        self.UNet = UNet()

        # -- initialize
        self._initialize_weights_randomly()

    def forward(self, audio_data, video_data, audio_lengths, video_lengths, audio_data_stft, audio_raw_data, boundaries=None):
        if self.modality == "av":
            
        
            # audio feature extraction
            # (B,1,18560)
            B, C, T = audio_data.size() ## audio is not normalized (-1,1)
            audio_lengths = [audio_lengths[0]]*B
            video_lengths = [video_lengths[0]]*B

            audio_data = self.audio_trunk(audio_data)
            audio_data = audio_data.transpose(1, 2) # (B, T, 512)
            audio_lengths = [_//640 for _ in audio_lengths] 

            audio_raw_data = self.audio_trunk(audio_raw_data.unsqueeze(1))
            audio_raw_data = audio_raw_data.transpose(1, 2) # (B, T, 512)
            audio_lengths = [_//640 for _ in audio_raw_data] 
            
            # video feature extraction
            # (B,1, 29,88,88)
            B, C, T, H, W = video_data.size()
            video_data = self.frontend3D(video_data)
            Tnew = video_data.shape[2]    # outpu should be B x C2 x Tnew x H x W
            video_data = threeD_to_2D_tensor(video_data)
            video_data = self.video_trunk(video_data) # (B, T, 512)

            video_data = video_data.view(B, Tnew, video_data.size(1))


            ## transformer seperator
            audio, video = self.seperator(audio_data,video_data) # B, T, 512
            # print("audio", audio[0])
            # av_feature = torch.cat([audio, video], dim=2)
            
            # real, imag = self.UNet(audio_data_stft.transpose(1, 3), av_feature.transpose(1, 2))

            return audio, audio_raw_data, audio_data, video
            # b 512 -> b 4096 -> b 64 64 -> b 32 32 16-> b 16 16 64 -> b 8 8 256 -> nn.pixelshuffle b 128 128

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))



def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'audio','av'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- DenseTCN config
    parser.add_argument('--densetcn-block-config', type=int, nargs = "+", help='number of denselayer for each denseTCN block')
    parser.add_argument('--densetcn-kernel-size-set', type=int, nargs = "+", help='kernel size set for each denseTCN block')
    parser.add_argument('--densetcn-dilation-size-set', type=int, nargs = "+", help='dilation size set for each denseTCN block')
    parser.add_argument('--densetcn-growth-rate-set', type=int, nargs = "+", help='growth rate for DenseTCN')
    parser.add_argument('--densetcn-dropout', default=0.2, type=float, help='Dropout value for DenseTCN')
    parser.add_argument('--densetcn-reduced-size', default=256, type=int, help='the feature dim for the output of reduce layer')
    parser.add_argument('--densetcn-se', default = False, action='store_true', help='If True, enable SE in DenseTCN')
    parser.add_argument('--densetcn-condense', default = False, action='store_true', help='If True, enable condenseTCN')
    # -- attention config
    parser.add_argument('--attention-embed-dim', type=int, default = 1664,  help='Attention layer input embedding size')
    parser.add_argument('--attention-num-head', type=int, default = 8, help='Attention layer head num')
    parser.add_argument('--attention-dropout', type=float, default = 0.2, help='Attention layer dropout')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default="./configs/lrw_resnet18_dctcn.json", help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')
    # use boundaries
    parser.add_argument('--use-boundary', default=False, action='store_true', help='include hard border at the testing stage.')
    # -- Spectrogram config
    parser.add_argument('--spectrogram-hop-length', type=int, default=145, help='hop length of spectrogram')
    parser.add_argument('--n-fft', type=int, default=256, help='n_fft for making spectrogram')
    parser.add_argument('--spectrogram-sample-rate', type=int, default=16000, help='sampling rate of spectrogram')

    # -- sample path
    parser.add_argument('--test-sample-path',type = str, default = './final.mp4', help = "path to save sample")
    parser.add_argument('--video-path',type=str,default='./test_video.mp4',help ="path to inference video")
    parser.add_argument('--transformer', default=False, action='store_true', help='use avsep ')

    parser.add_argument('--unet', default=False, action='store_true', help="use unet")

    # -- loss type
    parser.add_argument('--loss-type', default="coordinate", type = str, help="loss type : phase or coordinate")

    args = parser.parse_args()
    return args

args = load_args()
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
# device detection - cuda or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_num = torch.cuda.device_count()
#device = "cpu"


def audio_to_stft(batch_data, n_fft, hop_length, return_complex, sequence_len): ## returns short-time fourier transform value
    stft = torch.stft(batch_data, n_fft = n_fft, hop_length = hop_length,return_complex=return_complex, onesided=False)

    return stft[:,:n_fft // 2,:sequence_len]

def inference(model,video_data,audio_data):
    model.eval()
    audio_lengths =[18560]*(len(audio_data))
    video_lengths = [29]*(len(video_data))

    audio_data_stft = audio_to_stft(audio_data,256,145,True,128).to(device) 
    audio_data_angle = torch.angle(audio_data_stft).to(device) ## get angle from audio_data
    audio_data = audio_data.unsqueeze(1).to(device) 
    video_data = video_data.unsqueeze(1).to(device)
    #print(audio_raw_stft.shape)
    if args.unet:
        audio_data_stft = audio_to_stft(audio_data.squeeze(), 1024, 640, True, 29) # 
        logits = model(audio_data,video_data, audio_lengths,video_lengths,audio_data_stft)
    else:
        logits = model(audio_data,video_data, audio_lengths,video_lengths)
    if args.transformer:
        if args.loss_type == "phase":
            phase,amplitude = logits
            #amplitude = torch.max(torch.abs(audio_data_stft)) * amplitude ## rescale amplitdue to original 
            pred_mask = torch.polar(amplitude,phase * np.pi * 2)
            reconstructed_waveform = torch.istft(torch.cat([audio_data_stft*pred_mask.squeeze(), torch.zeros(pred_mask.size(0), 1, pred_mask.size(2)).to(device)], dim=1).to(device),n_fft=args.n_fft,hop_length=145)
            return reconstructed_waveform
        else:
            pred_real_mask, pred_imag_mask = logits
            noise_real, noise_imag = torch.real(audio_data_stft), torch.imag(audio_data_stft)

            pred_real, pred_imag = (noise_real * pred_real_mask) - (noise_imag * pred_imag_mask), (noise_real * pred_imag_mask) + (noise_imag * pred_real_mask)

            pred_out = torch.complex(pred_real, pred_imag)
            reconstructed_waveform = torch.istft(torch.cat([pred_out, torch.zeros(pred_out.size(0), 1, pred_out.size(2)).to(device)], dim=1).to(device),n_fft=1024,hop_length=640)
            return reconstructed_waveform
    reconstructed_waveform = torch.istft(torch.cat([logits*torch.exp(1j*audio_data_angle), torch.zeros(logits.size(0), 1, logits.size(2)).to(device)], dim=1).to(device),n_fft=args.n_fft,hop_length=145)
    #ouptut 
    # shell
    return reconstructed_waveform

def load_json( json_fp ):
    assert os.path.isfile( json_fp ), "Error loading JSON. File provided does not exist, cannot read: {}".format( json_fp )
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content

def get_model_from_json():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
    else:
        densetcn_options = {}

    if args.modality == "av": ## multi modal lipreading model
        attention_options = {
            'embed_dim' : args.attention_embed_dim,
            'num_heads' : args.attention_num_head,
            'dropout' : args.attention_dropout,
        }
        if args.unet:
            seperator_options={
                "d_model" :512,
                "n_head" : 8,
                "num_layers" : [1,2,3]
            }
            model = AVLipreading_sep_unet( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        attention_options=attention_options,
                        seperator_options = seperator_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        use_boundary=args.use_boundary,
                        extract_feats=args.extract_feats
                        )
            return model
        if args.transformer:
            seperator_options={
                "d_model" :512,
                "n_head" : 8,
                "num_layers" : [2,3]
            }
            model = AVLipreading_sep( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        attention_options=attention_options,
                        seperator_options = seperator_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        use_boundary=args.use_boundary,
                        extract_feats=args.extract_feats
                        )
            return model
        model = AVLipreading( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        attention_options=attention_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        use_boundary=args.use_boundary,
                        extract_feats=args.extract_feats)

        return model


def load_model(load_path, model, optimizer = None, allow_size_mismatch = False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile( load_path ), "Error when loading the model, provided path not found: {}".format( load_path )
    checkpoint = torch.load(load_path)
    loaded_state_dict = checkpoint['model_state_dict']
    # if allow_size_mismatch:
    #     loaded_sizes = { k: v.shape for k,v in loaded_state_dict.items() }
    #     model_state_dict = model.state_dict()
    #     model_sizes = { k: v.shape for k,v in model_state_dict.items() }
    #     mismatched_params = []
    #     for k in loaded_sizes:
    #         if loaded_sizes[k] != model_sizes[k]:
    #             mismatched_params.append(k)
    #     for k in mismatched_params:
    #         del loaded_state_dict[k]
    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict)
    return model

def save_video(video_path,audio_path,save_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    video = video.set_audio(audio)

    video.write_videofile(save_path,codec="libx264",audio_codec="aac")


def main():
    

    
    start = time.time()
    model = get_model_from_json()
    model = torch.nn.DataParallel(model)
    assert args.model_path, f"must specify model path."
    model = load_model(args.model_path,model,allow_size_mismatch=args.allow_size_mismatch)
    model.to(device)
    dset_loaders = get_data_loaders(args)['test']
    for batch_idx, data in enumerate(tqdm(dset_loaders)):
        audio_data,video_data,audio_lengths,video_lengths,audio_raw_data = data

        audio_lengths = [audio_lengths[0]]*(len(audio_lengths)//(gpu_num))
        video_lengths = [video_lengths[0]]*(len(video_lengths)//(gpu_num))
        #temp = mel_transform(audio_data.detach()) 
        
        audio_data = audio_data.unsqueeze(1).to(device) 
        video_data = video_data.unsqueeze(1).to(device)

        audio_raw_stft = audio_to_stft(audio_raw_data, 1024, 640, False, 29).to(device)
        audio_data_stft = audio_to_stft(audio_data.squeeze(), 1024, 640, True, 29)
        #print(audio_raw_stft.shape)
        separated_audio, raw_audio, noised_audio_data, separated_video  = model(audio_data,video_data, audio_lengths,video_lengths, audio_data_stft, audio_raw_data)

        reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.5, metric="euclidean")

        concated_feature = torch.cat([separated_audio[0], raw_audio[0], noised_audio_data[0]])

        embedding = reducer.fit_transform(concated_feature.detach().cpu().numpy())
        

        separated_audio_embedding = embedding[:29]
        print(separated_audio_embedding.shape)

        raw_audio_embedding = embedding[29:58]
        print(raw_audio_embedding.shape)

        noise_audio_embedding = embedding[58:]
        print(noise_audio_embedding.shape)

        concated_embedding = np.concatenate([separated_audio_embedding, raw_audio_embedding, noise_audio_embedding])

        print(concated_embedding.shape)

        types = ['separated_audio_embedding'] * 29 + ['raw_audio_embedding'] * 29 + ['noise_audio_embedding'] * 29  # Type labels for each data point

        # Define colors for each type
        type_colors = {'separated_audio_embedding': [0,0,0], 'raw_audio_embedding': [100,0,0], 'noise_audio_embedding':[255,0,0]}

        # Create a list of colors corresponding to each data point
        colors = np.array([type_colors[t] for t in types])

        # Create the scatter plot with different colors for each type
        vis.scatter(
            X=concated_embedding,
            opts=dict(
                title='Scatter Plot',
                markersize=10,
                xlabel='X',
                ylabel='Y',
                markercolor=colors,
                legend=list(type_colors.keys()),  # Add the legend
            )
        )


        print(separated_audio.shape)
        print(separated_video.shape)
        print(raw_audio.shape)

                
        return

    video_data,audio_data = video_preprocessing(args.video_path)
    reconstructed_waveform = inference(model,video_data,audio_data)
    #print(reconstructed_waveform.shape)
    #print(reconstructed_waveform)
    cat=reconstructed_waveform.reshape(1,-1)
    #print(cat)
    torchaudio.save('./sample.wav',cat.detach().cpu(),16000)
    save_video('./test_output.mp4','./sample.wav',args.test_sample_path)
    print("done!")
    # end = time.time()-  start
    # print(f'elapsed time={end}')
    # # #print(logits.shape)
    # print("inference finished")

    
    

if __name__ == "__main__":
    main()