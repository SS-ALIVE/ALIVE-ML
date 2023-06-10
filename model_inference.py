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
from lipreading.model import AVLipreading, AVLipreading_sep, AVLipreading_sep_unet
from moviepy.editor import VideoFileClip, AudioFileClip
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
#device = "cpu"


def video_preprocessing(video_path):

    input_file = video_path
    output_file = "./test_output.mp4"  # Path to the output video file
    desired_frame_rate = 25  # Desired frame rate for the output video

    # # FFmpeg command to change the frame rate
    ffmpeg_command = f'ffmpeg -i {input_file} -filter:v fps=25 {output_file} -y' ## frame conversion

    # # Execute the FFmpeg command
    subprocess.call(ffmpeg_command, shell=True)

    ## face detector
    detector = dlib.get_frontal_face_detector()

    video = cv2.VideoCapture('./test_output.mp4')
    original_fps = video.get(cv2.CAP_PROP_FPS)

    audio_data = librosa.load('./test_output.mp4',sr=16000)[0] ## load audio
    #print(audio_data.shape)

    #video.set(cv2.CAP_PROP_FPS, 240) ## set fps as 30
    roi_w,roi_h = 0,0
    video_frames=deque()
    frame_count=0
    face_count = 0
    while True:
        # Read a frame
        print(f'frame_count={len(video_frames)}, processing..')
        # Break the loop if the video is finished
        ret, frame = video.read()
        frame_count+=1
        if not ret:
            break
        # Convert the frame to grayscale for face detection
        #frame = cv2.resize(frame,(500,500))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = detector(frame)
        if len(faces)==0:
            video_frames.append(cv2.resize(frame,(88,88)))
            continue
        if roi_w==0 and roi_h==0:
            x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
            roi_w = w 
            roi_h = h
        # Iterate through the detected faces
        for face in faces:
            face_count+=1
            # Extract the mouth region from the face bounding box
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            roi_x = int(x + w / 2 - roi_w / 2)
            roi_y = int(y + h / 2 - roi_h / 2)
            roi_width = roi_w
            roi_height = roi_h

            # Ensure the ROI coordinates are within the frame boundaries
            roi_x = max(roi_x, 0)
            roi_y = max(roi_y, 0)
            roi_x_end = min(roi_x + roi_width, frame.shape[1])
            roi_y_end = min(roi_y + roi_height, frame.shape[0])

            # Extract the face ROI from the frame
            face_roi = frame[roi_y:roi_y_end, roi_x:roi_x_end]
            face_roi = cv2.resize(face_roi,(88,88))
            video_frames.append(face_roi)
            # Display the face ROI
            #cv2.imshow('Face ROI', face_roi)
        
            
            # Draw a rectangle around the face in the frame
            #cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        # Display the frame
        #cv2.imshow('Mouth ROI', frame)

        # Exit the loop if the 'q' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    #print(video_frame.shape)
    #print(audio_data.shape) ## [291643] --> [,16000] / 15*(640*29) + (640*20)
    #print(video_frame)
    # Release the video capture and close all windows
    video.release()
    cv2.destroyAllWindows()
    print("nimyeonsang!=",face_count)

    #make video processable (batch)
    video_frames = torch.tensor(np.array(video_frames)) ## [455,96,96] 
    video_frames = (video_frames) / 255.0
    mean = torch.mean(video_frames)
    std = torch.std(video_frames)
    normalized_video_frames = (video_frames - mean)/ std
    batch_len = (len(video_frames)-1)//29 + 1
    vid_pad_len = batch_len*29-len(video_frames)
    video_padding = torch.zeros(vid_pad_len,88,88)
    video_data = torch.cat((normalized_video_frames,video_padding)).reshape(batch_len,29,88,88)
    
    aud_pad_len = batch_len*18560 - len(audio_data)
    audio_padding = torch.zeros(aud_pad_len)
    audio_data = torch.cat((torch.tensor(audio_data),audio_padding)).reshape(batch_len,18560) 

    return video_data,audio_data ## return processed video,audio data 


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
                "num_layers" : [2,3]
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
    print("model loaded!")
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