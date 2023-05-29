import torch
import torch.nn as nn
import math
import numpy as np
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.resnet1D import ResNet1D, BasicBlock1D
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from lipreading.models.densetcn import DenseTemporalConvNet
from lipreading.models.swish import Swish
from lipreading.models.FCN import FCN
import torchaudio.transforms as transforms
from lipreading.dataset import audio_to_stft

# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B): # 
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )

def _transposed_average_batch(x,lengths,B):
    return torch.stack([torch.mean(x[index][0:i,:],0) for index,i in enumerate(lengths)],0)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func( out, lengths, B )
        return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)


class DenseTCN(nn.Module):
    def __init__( self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type,
                  squeeze_excitation=False,
        ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet( block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)

class DenseTCN_feature(nn.Module):
    def __init__( self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type,
                  squeeze_excitation=False,
        ):
        super(DenseTCN_feature, self).__init__()

        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet( block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        # self.tcn_output = nn.Linear(num_features, num_classes)
        # self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        # x = self.consensus_func( x, lengths, B )
        # return self.tcn_output(x)
        return x

class AVCrosssAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(AVCrosssAttention,self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.a_self_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads = self.num_heads, dropout = self.dropout ,batch_first=True)
        self.v_self_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads = self.num_heads, dropout = self.dropout ,batch_first=True)
        self.av_cross_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads = self.num_heads, dropout = self.dropout ,batch_first=True) # cross-modality feature
        self.va_cross_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads = self.num_heads, dropout = self.dropout ,batch_first=True)

    def forward(self,audio_feature,video_feature):
        a_self_attention = self.a_self_attention(audio_feature,audio_feature,audio_feature)[0]
        v_self_attention = self.v_self_attention(video_feature,video_feature,video_feature)[0]
        av_attention = self.av_cross_attention(video_feature, audio_feature, video_feature)[0] # return attention value
        va_attention = self.va_cross_attention(audio_feature, video_feature, audio_feature)[0] 
        return av_attention, va_attention, a_self_attention,v_self_attention


class Lipreading(nn.Module):
    def __init__( self, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500,
                  relu_type='prelu', tcn_options={}, densetcn_options={}, width_mult=1.0,
                  use_boundary=False, extract_feats=False):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary

        if self.modality == 'audio':
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
        elif self.modality == 'video':
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            elif self.backbone_type == 'shufflenet':
                assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
                shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
                self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
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

        if tcn_options:
            tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
            self.tcn = tcn_class( input_size=self.backend_out,
                                  num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                                  num_classes=num_classes,
                                  tcn_options=tcn_options,
                                  dropout=tcn_options['dropout'],
                                  relu_type=relu_type,
                                  dwpw=tcn_options['dwpw'],
                                )
        elif densetcn_options:
            self.tcn =  DenseTCN( block_config=densetcn_options['block_config'],
                                  growth_rate_set=densetcn_options['growth_rate_set'],
                                  input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                                  reduced_size=densetcn_options['reduced_size'],
                                  num_classes=num_classes,
                                  kernel_size_set=densetcn_options['kernel_size_set'],
                                  dilation_size_set=densetcn_options['dilation_size_set'],
                                  dropout=densetcn_options['dropout'],
                                  relu_type=relu_type,
                                  squeeze_excitation=densetcn_options['squeeze_excitation'],
                                )
        else:
            raise NotImplementedError

        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, x, lengths, boundaries=None):
        if self.modality == 'video':
            B, C, T, H, W = x.size()
            x = self.frontend3D(x)
            Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor( x )
            x = self.trunk(x)

            if self.backbone_type == 'shufflenet':
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))
        elif self.modality == 'audio':
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_//640 for _ in lengths]


        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)

        return x if self.extract_feats else self.tcn(x, lengths, B)


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

class AVLipreading(nn.Module): ## new model - audio-visual cross attention
    def __init__( self, modality='av', hidden_dim=256, backbone_type='resnet', num_classes=500,
                  relu_type='prelu', tcn_options={}, densetcn_options={}, attention_options = {}, width_mult=1.0,
                  use_boundary=False, extract_feats=False):
        super(AVLipreading, self).__init__()
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

        ## cross- model attention module
        
        if tcn_options:
            tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
            self.audio_tcn = tcn_class( input_size=self.backend_out,
                                  num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                                  num_classes=num_classes,
                                  tcn_options=tcn_options,
                                  dropout=tcn_options['dropout'],
                                  relu_type=relu_type,
                                  dwpw=tcn_options['dwpw'],
                                )
            self.video_tcn = tcn_class( input_size=self.backend_out,
                                  num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                                  num_classes=num_classes,
                                  tcn_options=tcn_options,
                                  dropout=tcn_options['dropout'],
                                  relu_type=relu_type,
                                  dwpw=tcn_options['dwpw'],
                                )

        elif densetcn_options:
            self.audio_tcn =  DenseTCN_feature( block_config=densetcn_options['block_config'], 
                                  growth_rate_set=densetcn_options['growth_rate_set'],
                                  input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                                  reduced_size=densetcn_options['reduced_size'],
                                  num_classes=num_classes,
                                  kernel_size_set=densetcn_options['kernel_size_set'],
                                  dilation_size_set=densetcn_options['dilation_size_set'],
                                  dropout=densetcn_options['dropout'],
                                  relu_type=relu_type,
                                  squeeze_excitation=densetcn_options['squeeze_excitation'],
                                )

            self.video_tcn =  DenseTCN_feature( block_config=densetcn_options['block_config'], 
                                  growth_rate_set=densetcn_options['growth_rate_set'],
                                  input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                                  reduced_size=densetcn_options['reduced_size'],
                                  num_classes=num_classes,
                                  kernel_size_set=densetcn_options['kernel_size_set'],
                                  dilation_size_set=densetcn_options['dilation_size_set'],
                                  dropout=densetcn_options['dropout'],
                                  relu_type=relu_type,
                                  squeeze_excitation=densetcn_options['squeeze_excitation'],
                                )

        else:
            raise NotImplementedError

        self.cross_attention = AVCrosssAttention(
            embed_dim = attention_options['embed_dim'], 
            num_heads = attention_options['num_heads'] , 
            dropout = attention_options['dropout'],
            )
        self.consensus_func = _transposed_average_batch

        # self.mel_transform = transforms.MelSpectrogram(
        #         sample_rate = 16000,
        #         n_fft=1024,
        #         hop_length=145,
        #         n_mels=128,
        #         norm='slaney'
        #     )

        self.spec_transform = audio_to_stft

        self.FCN = FCN(feature_dim = 2048) ##TODO need to specify how to pass feature dimension argument 

        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, audio_data, video_data, audio_lengths, video_lengths, boundaries=None):
        if self.modality == "av":

        
            # audio feature extraction
            # (B,1,18560)
            B, C, T = audio_data.size() ## audio is not normalized (-1,1)
            audio_lengths = [audio_lengths[0]]*B
            video_lengths = [video_lengths[0]]*B
            #print("audio_data - max",torch.max(audio_data))
            ## mel-spectogram generation
            # audio_data = (B,18560),normalized tensor #TODO : must isolate mel-spectogram generation from forward computation
            #print(torch.max(audio_data))
            
            stft = self.spec_transform(audio_data.squeeze())
            spectrogram = torch.abs(stft)
            #mel = self.mel_transform(audio_data.squeeze()) # generated mel-spectogram
            #print(torch.max(mel_spec))
            #print(mel_spec.shape)
            ###
            
            # (b,1,18560) (32,1,18560)
            # audio_data = audio_data.squeeze() ## normalize
            # mean = torch.mean(audio_data,dim=1,keepdim=True)
            # std = torch.std(audio_data,dim=1,keepdim=True)
            # audio_data = (audio_data - mean)/std 
            # audio_data = audio_data.unsqueeze(1)
            
            #print(torch.max(audio_data))
            audio_data = self.audio_trunk(audio_data)
            audio_data = audio_data.transpose(1, 2) # (B, T, 512)
            audio_lengths = [_//640 for _ in audio_lengths] 
            
            # video feature extraction
            # (B,1, 29,88,88)
            B, C, T, H, W = video_data.size()
            video_data = self.frontend3D(video_data)
            Tnew = video_data.shape[2]    # outpu should be B x C2 x Tnew x H x W
            video_data = threeD_to_2D_tensor(video_data)
            video_data = self.video_trunk(video_data) # (B, T, 512)

            video_data = video_data.view(B, Tnew, video_data.size(1))
            
            audio_data = self.audio_tcn(audio_data, audio_lengths, B) # (B, T, 1664)  1664 -> 512 + 384 + 384 + 384 // 512
            video_data = self.video_tcn(video_data, video_lengths, B) # (B, T, 1664)  1664 -> 512 + 384 + 384 + 384 // 512
            
            audio_data = audio_data.transpose(1,2)
            video_data = video_data.transpose(1,2)

            av_attention, va_attention,a_self_attention,v_self_attention = self.cross_attention(audio_data, video_data) # self attention & cross-attention
            #print("av_attention shape = ",av_attention.shape)
            #print("va_attention shape = ",va_attention.shape)
            a_self_attention = self.consensus_func(a_self_attention,audio_lengths,B)
            v_self_attention = self.consensus_func(v_self_attention,video_lengths,B)
            av_attention = self.consensus_func(av_attention,audio_lengths,B) # temporal averaging -> (B,T,F) - > (B,F)
            va_attention = self.consensus_func(va_attention,video_lengths,B)  #
            ava_attention = torch.cat((av_attention,va_attention,a_self_attention,v_self_attention),1) ## concatenate audio-visua, visual-audio attention (B,2*F)
            ## FCN layer - generates Mask to apply with original audio's mel-spectogram
            mask = self.FCN(ava_attention) ## reconstruct mel-spectogram mask with U-NET : (B,1664) -> (B,1,128,128)
            #mel-spectogram..
            mask = mask.squeeze()
            # print("mask = ",mask.shape)
            # print("raw = ",mel.shape)
            
            out = torch.mul(mask,spectrogram) # element-wise multiplication (mask, original) - mel-spectogram

            # print("out", out.shape)
            
        return out


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
        