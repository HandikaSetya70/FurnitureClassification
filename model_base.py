import torch
import torch.nn as nn

class SimpleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class SimpleNet2D(nn.Module):
    def __init__(self, output_class: int) -> None:
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Downsampling blocks
        self.block1 = SimpleBlock(64, 128)    # 177 -> 88
        self.block2 = SimpleBlock(128, 256)   # 88 -> 44
        self.block3 = SimpleBlock(256, 512)   # 44 -> 22
        self.block4 = SimpleBlock(512, 512)   # 22 -> 11
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_class)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)          # N, 64, 177, 177
        
        # Downsampling path
        x = self.block1(x)         # N, 128, 88, 88
        x = self.block2(x)         # N, 256, 44, 44
        x = self.block3(x)         # N, 512, 22, 22
        x = self.block4(x)         # N, 512, 11, 11
        
        # Global pooling
        x = self.global_pool(x)    # N, 512, 1, 1
        
        # Reshape
        x = x.view(x.size(0), -1)  # N, 512
        
        # Classification
        x = self.classifier(x)     # N, num_classes
        
        return x
class WebCompatibleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # Using stride in Conv2d instead of separate pooling
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.downsample(x)
        return x

class WebCompatibleNet(nn.Module):
    def __init__(self, output_class: int) -> None:
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Downsampling blocks
        self.block1 = WebCompatibleBlock(64, 128)    # 177 -> 88
        self.block2 = WebCompatibleBlock(128, 256)   # 88 -> 44
        self.block3 = WebCompatibleBlock(256, 512)   # 44 -> 22
        self.block4 = WebCompatibleBlock(512, 512)   # 22 -> 11
        
        # Using Conv2d for global pooling
        self.global_pool = nn.Conv2d(512, 512, kernel_size=11)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_class)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)          # N, 64, 177, 177
        
        # Downsampling path
        x = self.block1(x)         # N, 128, 88, 88
        x = self.block2(x)         # N, 256, 44, 44
        x = self.block3(x)         # N, 512, 22, 22
        x = self.block4(x)         # N, 512, 11, 11
        
        # Global pooling
        x = self.global_pool(x)    # N, 512, 1, 1
        
        # Reshape
        x = x.view(x.size(0), -1)  # N, 512
        
        # Classification
        x = self.classifier(x)     # N, num_classes
        
        return x

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
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
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

# For testing
if __name__ == "__main__":
    # Test the model
    model = SimpleNet2D(7)  # 7 classes for your furniture dataset
    x = torch.randn(1, 3, 177, 177)  # Test input
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")