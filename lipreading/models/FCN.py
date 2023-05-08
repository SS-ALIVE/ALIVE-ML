import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channel):

        super(FCN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.segment = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.block1(x)
        x1 = x
        # print("x1 : ", x1.shape)
        x = self.block2(x)
        x2 = x
        # print("x2 : ", x2.shape)
        x = self.block3(x)
        x3 = x
        # print("x3 : ", x3.shape)
        x = self.block4(x)
        x4 = x
        # print("x4 : ", x4.shape)
        x = self.block5(x)
        x5 = x
        # print("x5 : ", x5.shape)
        
        feature = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        # print(feature.shape)
        feature = feature + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        feature = self.bn2(self.relu(self.deconv2(feature)))  # size=(N, 256, x.H/8, x.W/8)
        # print(feature.shape)
        feature = feature + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        feature = self.bn3(self.relu(self.deconv3(feature)))  # size=(N, 128, x.H/4, x.W/4)
        # print(feature.shape)
        feature = feature + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        feature = self.bn4(self.relu(self.deconv4(feature)))  # size=(N, 64, x.H/2, x.W/2)
        # print(feature.shape)
        mask    = self.sigmoid(self.segment(feature))
        
        return mask
    

