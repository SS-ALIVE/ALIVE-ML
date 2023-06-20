import torch
import torch.nn as nn
from lipreading.models.swish import Swish

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.block1 = nn.Sequential( # 1/2 H 1/2 W 2 C
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            Swish(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.block2 = nn.Sequential( # 1/8 H 1/8 W 8 C
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            Swish(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.block3 = nn.Sequential( # 1/32 H 1/32 W 32 C
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            Swish(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.block4 = nn.Sequential( # 1/128 H 1/128 W 128 C
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Swish(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.block5 = nn.Sequential( # 1/512 H 1/512 W 512 C
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            Swish(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        self.swish    = Swish()

        self.deconv5 = nn.ConvTranspose2d(1024 + 1024, 256, kernel_size=(1, 4), stride=(1, 4))
        self.bn5     = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256 + 256, 64, kernel_size=(1, 4), stride=(1, 4))
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64 + 64, 16, kernel_size=(1, 4), stride=(1, 4))
        self.bn3     = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16 + 16, 4, kernel_size=(1, 4), stride=(1, 4))
        self.bn2     = nn.BatchNorm2d(4)
        self.deconv1 = nn.ConvTranspose2d(4 + 4, 2, kernel_size=(1, 2), stride=(1, 2))
        self.bn1     = nn.BatchNorm2d(2)
        
        self.sigmoid = nn.Tanh()



    def forward(self, stft, av_feature):
        encoded_1 = self.block1(stft) # 29 128 2 -> 29 64 4
        
        encoded_2 = self.block2(encoded_1) # 29 64 4 -> 29 32 8
        
        encoded_3 = self.block3(encoded_2) # 29 32 8 -> 29 16 16
        
        encoded_4 = self.block4(encoded_3) # 29 16 16 -> 29 4 64
        
        encoded_5 = self.block5(encoded_4) # 29 4 64 -> 29 1 256
        
        # av_feature = torch.zeros_like(av_feature)
        hidden_vector = torch.cat([encoded_5, av_feature.unsqueeze(-1)], dim=1)

        
        decoded_5 = self.bn5(self.swish(self.deconv5(hidden_vector))) # 29 1 256+1024 -> 29 4 64
        
        decoded_4 = self.bn4(self.swish(self.deconv4(torch.cat([decoded_5, encoded_4], dim=1)))) # 29 4 64+64 -> 29 16 16
        
        decoded_3 = self.bn3(self.swish(self.deconv3(torch.cat([decoded_4, encoded_3], dim=1)))) # 29 4 16+16 -> 29 32 8
        
        decoded_2 = self.bn2(self.swish(self.deconv2(torch.cat([decoded_3, encoded_2], dim=1)))) # 29 32 8+8 -> 29 64 4
        
        decoded_1 = self.bn1(self.swish(self.deconv1(torch.cat([decoded_2, encoded_1], dim=1)))) # 29 64 4+4 -> 29 128 2
        
        stft_mask = decoded_1
        # stft_mask = self.sigmoid(decoded_1)
        
        
        return stft_mask[:, 0], stft_mask[:, 1]
    

