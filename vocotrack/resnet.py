import torch
import torch.nn as nn
from typing import ClassVar
from .config import ResNetCfg, BlockType


class _ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: int,
        padding: int | None = None
    ):
        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )


class _ConvBNRelu(_ConvBN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: int,
        padding: int | None = None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.add_module('relu', nn.ReLU(inplace=True))


class _ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        conv_block: nn.Sequential
    ):
        super().__init__()
        self.conv_block = conv_block
        self.identity = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.identity = _ConvBN(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)
        out = self.conv_block(x)
        return self.relu(identity + out)


class _Bottleneck(_ResBlock):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        bott_channels: int,
        stride: int = 1
    ):
        out_channels = bott_channels * self.expansion

        conv_block = nn.Sequential(
            _ConvBNRelu(in_channels, out_channels, 1, stride=1, padding=0),
            _ConvBNRelu(bott_channels, bott_channels, 3, stride=stride),
            _ConvBN(bott_channels, out_channels, 1, stride=1, padding=0)
        )

        super().__init__(in_channels, out_channels, stride, conv_block)


class _Basic(_ResBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        conv_block = nn.Sequential(
            _ConvBNRelu(in_channels, out_channels, 3, stride),
            _ConvBN(out_channels, out_channels, 3, stride=1)
        )

        super().__init__(in_channels, out_channels, stride, conv_block)


class _Stage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        stride: int,
        blocks: int,
        block_type: BlockType,
    ):
        if block_type is BlockType.BASIC:
            layers = [_Basic(in_channels, num_channels, stride)]
            out_channels = num_channels
            def factory(): return _Basic(num_channels, num_channels)
        else:
            layers = [_Bottleneck(in_channels, num_channels, stride)]
            out_channels = _Bottleneck.expansion * num_channels
            def factory(): return _Bottleneck(out_channels, num_channels)

        for _ in range(1, blocks):
            layers.append(factory())

        self._out_channels = out_channels
        super().__init__(*layers)

    @property
    def out_channels(self) -> int:
        return self._out_channels


class ResNet(nn.Module):
    def __init__(self, config: ResNetCfg):
        super().__init__() 
        config = ResNetCfg.model_validate(config)
        
        self.stem = _ConvBNRelu(**config.stem.model_dump())
        self.model = nn.Sequential(
            *(stages := [
                _Stage(
                    stage.in_channels,
                    stage.num_channels,
                    stage.stride,
                    stage.blocks,
                    config.block_type
                )
                for stage in config.stages
            ])
        )

        self._out_channels = stages[-1].out_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.stem(x)
        return self.model(x)
