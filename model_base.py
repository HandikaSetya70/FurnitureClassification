import torch
import torch.nn as nn

class BasicConvolution(nn.Module):
    def __init__(self, input_ch: int, output_ch: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(input_ch, output_ch, kernel_size=3, padding='same')
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class BasicConvBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int) -> None:
        super().__init__()
        self.conv = BasicConvolution(input_ch, output_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_class: int) -> None:
        super().__init__()

        # 3, 177, 177
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=7, padding="same")
        self.block_1 = BasicConvBlock(16, 32)
        
        # 32, 88, 88
        self.block_2 = BasicConvBlock(32, 64)
        
        # 64, 44, 44
        self.block_3 = BasicConvBlock(64, 128)

        # 128, 22, 22
        self.block_4 = BasicConvBlock(128, 256)

        # 256, 11, 11
        self.pool = nn.AdaptiveMaxPool2d(1)

        # 256, 1, 1
        # RESHAPE 
        # 256

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_class)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)   # (N,   3, 177, 177) => (N,  16, 177, 177)
        x = self.block_1(x)  # (N,  16, 177, 177) => (N,  32,  88,  88)
        x = self.block_2(x)  # (N,  32,  88,  88) => (N,  64,  44,  44)
        x = self.block_3(x)  # (N,  64,  44,  44) => (N, 128,  22,  22)
        x = self.block_4(x)  # (N, 128,  22,  22) => (N, 256,  11,  11)
        x = self.pool(x)     # (N, 256,  11,  11) => (N, 256,   1,   1)
        
        bz = x.size(0)       # batch size
        cz = x.size(1)       # channel size 
        x = x.view(bz, cz)   # (N, 256,   1,   1) => (N, 256)
        x = self.head(x)     # (N, 256) => (N, output_class)
        return x