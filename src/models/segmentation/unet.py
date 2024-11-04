import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__(EncoderBlock)
        
class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__(DecoderBlock)

class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__(UNet)
        
if __name__ == "__main__":
    x = torch.randn(572, 572, 3)
    print(x.shape)
        