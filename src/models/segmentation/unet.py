import torch
import torch.nn as nn

# typing
from typing import Union, Optional, Tuple, List

# dynamo
import torch._dynamo

class ConvSeq(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        padding: Union[str, int, Tuple[int, int]] = 1, 
        **kwargs
    ) -> None:
        super().__init__()
        
        self.conv_seq = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                padding=padding,
                **kwargs
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.conv_seq(x)
    
class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        skip_channels: int,
        out_channels: int,
        output_padding: int = 0,
    ) -> None:
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=output_padding
        )
        
        self.double_conv = nn.Sequential(
            ConvSeq(in_channels=out_channels + skip_channels, out_channels=out_channels, kernel_size=3),
            ConvSeq(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )
    
    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        x = self.deconv(x)
        x = torch.cat([x, cache], axis=1)
        x = self.double_conv(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        
        self.double_conv = nn.Sequential(
            ConvSeq(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            ConvSeq(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )
        self.down_sample_channels = [
            [64, 128],
            [128, 256],
            [256, 512],
            [512, 512]
        ]
        self.down_sample_list = nn.ModuleList(
            [self.down_sample(c[0], c[1]) for c in self.down_sample_channels]
        )
    
    def down_sample(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvSeq(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            ConvSeq(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor):
        cache = []
        x = self.double_conv(x)

        for downsample in self.down_sample_list:
            cache.append(x)
            x = downsample(x)
        
        return x, cache
        
class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.up_sample_params = [
            {'in_channels': 512, 'skip_channels': 512, 'out_channels': 256, 'output_padding': 1},
            {'in_channels': 256, 'skip_channels': 256, 'out_channels': 128, 'output_padding': 1},
            {'in_channels': 128, 'skip_channels': 128, 'out_channels': 64, 'output_padding': 0},
            {'in_channels': 64, 'skip_channels': 64, 'out_channels': 64, 'output_padding': 0},
        ]
        
        self.up_sample_list = nn.ModuleList(
            [UpSample(**params) for params in self.up_sample_params]
        )
        
    def forward(self, x: torch.Tensor, cache: List[torch.Tensor]): 
        for up_sample, cached_feature in zip(self.up_sample_list, reversed(cache)):
            x = up_sample(x, cached_feature)
        
        return x
    
class UNet(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.encode = EncoderBlock(in_channels=in_channels)
        self.decode = DecoderBlock()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    @staticmethod
    def build(num_classes: int, in_channels: int = 3, pretrained_model: Optional[str] = None) -> "UNet":
        model = UNet(num_classes=num_classes, in_channels=in_channels)
        
        if pretrained_model:
            try:
                model.init_weights()
            except (FileNotFoundError, RuntimeError) as e:
                print(e)
        
        return model
    
    def init_weights(self) -> None:
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x: torch.Tensor):
        x, cache = self.encode(x)
        x = self.decode(x, cache)
        logit = self.conv(x)
        return [logit]
        
if __name__ == "__main__":
    debug = False
    explain = False

    torch._dynamo.config.verbose = True
    if debug:
        torch._dynamo.config.log_level = "DEBUG"
    
    x = torch.randn(1, 3, 572, 572)  
    model = UNet.build(num_classes=2, in_channels=3)
    compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    output = compiled_model(x)
    print("Output shape:", output[0].shape)
    
    if explain:
        explanation = torch._dynamo.explain(compiled_model, x)
        print(explanation)
