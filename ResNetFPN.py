import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: A helper function to initialize weights with He normal initialization.
def init_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# -----------------------
# Residual Block (Encoder building block)
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), dropout=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act1  = activation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act2  = activation
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gn_sc = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.final_act = activation
        
        self.apply(init_he)
        
    def forward(self, x):
        shortcut = self.gn_sc(self.shortcut(x))
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act2(out)
        out = out + shortcut
        out = self.dropout(out)
        out = self.final_act(out)
        return out

# -----------------------
# ResNetFPN Model
# -----------------------
class ResNetFPN(nn.Module):
    def __init__(self, in_channels=3, n_filters=32, activation=nn.ReLU(inplace=True)):
        """
        Args:
            in_channels: Number of channels in the input image.
            n_filters: Base number of filters for the encoder.
            fpn_channels: Number of channels for each pyramid level.
            activation: Activation function.
        """
        super(ResNetFPN, self).__init__()
        self.activation = activation
        
        # Encoder path (using only the encoder from the UNet)
        self.c1 = ResidualBlock(in_channels, n_filters, activation=activation)
        self.pool1 = nn.MaxPool2d(2)
        self.c2 = ResidualBlock(n_filters, n_filters*2, activation=activation)
        self.pool2 = nn.MaxPool2d(2)
        self.c3 = ResidualBlock(n_filters*2, n_filters*4, activation=activation)
        self.pool3 = nn.MaxPool2d(2)
        self.c4 = ResidualBlock(n_filters*4, n_filters*8, activation=activation)
        self.pool4 = nn.MaxPool2d(2)
        self.c5 = ResidualBlock(n_filters*8, n_filters*16, activation=activation)
        
        self.layer5_outconv = conv1x1(n_filters*16, n_filters*16)

        self.layer4_outconv = conv1x1(n_filters*8, n_filters*16)
        self.layer4_outconv2 = nn.Sequential(
            conv3x3(n_filters*16, n_filters*16),
            nn.GroupNorm(num_groups=32, num_channels=n_filters*16),
            nn.LeakyReLU(),
            conv3x3(n_filters*16, n_filters*8),
        )

        self.layer3_outconv = conv1x1(n_filters*4, n_filters*8)
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(n_filters*8, n_filters*8),
            nn.GroupNorm(num_groups=32, num_channels=n_filters*8),
            nn.LeakyReLU(),
            conv3x3(n_filters*8, n_filters*4),
        )

        self.layer2_outconv = conv1x1(n_filters*2, n_filters*4)
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(n_filters*4, n_filters*4),
            nn.GroupNorm(num_groups=32, num_channels=n_filters*4),
            nn.LeakyReLU(),
            conv3x3(n_filters*4, n_filters*2),
        )

        self.layer1_outconv = conv1x1(n_filters, n_filters*2)
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(n_filters*2, n_filters*2),
           nn.GroupNorm(num_groups=32, num_channels=n_filters*2),
            nn.LeakyReLU(),
            conv3x3(n_filters*2, n_filters),
        )
        
        self.apply(init_he)
        
    def forward(self, x):
        # Bottom-up pathway (encoder)

        x1 = self.c1(x)            # (B, n_filters, H, W)
        p1 = self.pool1(x1)        # (B, n_filters, H/2, W/2)

        x2 = self.c2(p1)           # (B, n_filters*2, H/2, W/2)
        p2 = self.pool2(x2)        # (B, n_filters*2, H/4, W/4)

        x3 = self.c3(p2)           # (B, n_filters*4, H/4, W/4)
        p3 = self.pool3(x3)        # (B, n_filters*4, H/8, W/8)

        x4 = self.c4(p3)           # (B, n_filters*8, H/8, W/8)
        p4 = self.pool4(x4)        # (B, n_filters*8, H/16, W/16)
        # Bottleneck
        x5 = self.c5(p4)           # (B, n_filters*16, H/16, W/16)
        
        x5_out = self.layer5_outconv(x5) # n_filters*16

        x5_out_2x = F.interpolate(x5_out, scale_factor=2., mode='bilinear', align_corners=True)
        x4_out = self.layer4_outconv(x4)
        x4_out = self.layer4_outconv2(x4_out+x5_out_2x)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return x4_out, x1_out

# Example usage:
if __name__ == "__main__":
    model = ResNetFPN(in_channels=3, n_filters=32)
    x = torch.randn(4, 3, 192, 192)
    deep_feature, shallow_feature = model(x)

    print(deep_feature.shape)
    print(shallow_feature.shape)
    
