import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self,feature_dim):

        super(FCN, self).__init__()
        self.linear = nn.Linear(feature_dim,32*32)
        self.block1 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=3, padding=1),
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
        self.relu = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6     = nn.BatchNorm2d(16)
        self.deconv7 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1,1,32,32) # b,1024 -> b,1,32,32
        x = self.block1(x)
        x1 = x # b,64,16,16
        # print("x1 : ", x1.shape)
        x = self.block2(x)
        x2 = x #b,128,8,8
        # print("x2 : ", x2.shape)
        x = self.block3(x) # b,256,4,4
        x = self.bn3(self.relu(self.deconv3(x)))  # size=(N, 128, x.H/4, x.W/4)
        # print(feature.shape) # b,128,8,8
        x = x + x2                                 # element-wise add, size=(N, 128, x.H/4, x.W/4)
        x = self.bn4(self.relu(self.deconv4(x)))  # size=(N, 64, x.H/2, x.W/2)
        # print(feature.shape) # b,64,16,16
        x = x + x1
        x = self.bn5(self.relu(self.deconv5(x))) # b,32,32,32
        x = self.bn6(self.relu(self.deconv6(x))) # b,16,64,64
        mask= self.sigmoid(self.deconv7(x))
        
        return mask
    

