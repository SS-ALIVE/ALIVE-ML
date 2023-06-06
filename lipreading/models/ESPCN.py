import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self,feature_dim):

        super(ESPCN, self).__init__() # 64 64 -> 32 32 32 
        self.linear = nn.Linear(feature_dim,64*64)
        self.block1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16,32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential( # 32 32 32 -> 16 16 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential( ## 16 16 128 -> 8 8 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.pixelshuffle = nn.PixelShuffle(16)
    

    def forward(self, x):
        x = self.linear(x) # 
        x = x.view(-1,1,64,64) # b,1024 -> b,1,32,32
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.Tanh(x)
        x = self.pixelshuffle(x)
        x = self.sigmoid(x)
        return x
    

