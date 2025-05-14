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
        conv_block: nn.Sequential,
        dropout_prob: float = 0.2
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
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)
        out = self.conv_block(x)
        out = identity + out
        out = self.relu(out)
        out = self.dropout(out)
        return out


class _Bottleneck(_ResBlock):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        bott_channels: int,
        stride: int = 1,
        dropout_prob: float = 0.2
    ):
        out_channels = bott_channels * self.expansion

        conv_block = nn.Sequential(
            _ConvBNRelu(in_channels, out_channels, 1, stride=1, padding=0),
            _ConvBNRelu(bott_channels, bott_channels, 3, stride=stride),
            _ConvBN(bott_channels, out_channels, 1, stride=1, padding=0)
        )

        super().__init__(in_channels, out_channels, stride, conv_block, dropout_prob)


class _Basic(_ResBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_prob: float = 0.2
    ):
        conv_block = nn.Sequential(
            _ConvBNRelu(in_channels, out_channels, 3, stride),
            _ConvBN(out_channels, out_channels, 3, stride=1)
        )

        super().__init__(in_channels, out_channels, stride, conv_block, dropout_prob)

class _Stage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        stride: int,
        blocks: int,
        block_type: BlockType,
        dropout_prob: float = 0.2
    ):
        layers: list[nn.Module] = []
        if block_type is BlockType.BASIC:
            layers.append(_Basic(in_channels, num_channels, stride, dropout_prob=dropout_prob))
            out_channels = num_channels
            def factory(): return _Basic(num_channels, num_channels, dropout_prob=dropout_prob)
        else:
            layers.append(_Bottleneck(in_channels, num_channels, stride, dropout_prob=dropout_prob))
            out_channels = _Bottleneck.expansion * num_channels
            def factory(): return _Bottleneck(out_channels, num_channels, dropout_prob=dropout_prob)

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
        dropout_prob = config.dropout_prob
        
        self.stem = _ConvBNRelu(**config.stem.model_dump())
        self.dropout_stem = nn.Dropout2d(dropout_prob)
        
        self.model = nn.Sequential(
            *(stages := [
                _Stage(
                    stage.in_channels,
                    stage.num_channels,
                    stage.stride,
                    stage.blocks,
                    config.block_type,
                    dropout_prob=dropout_prob
                )
                for stage in config.stages
            ])
        )

        self._out_channels = stages[-1].out_channels
        
        total = config.stem.stride
        for stage_cfg in config.stages:
            total *= stage_cfg.stride
        self._total_stride = total

    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def total_stride(self) -> int:
        return self._total_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.dropout_stem(x)
        x = self.model(x)
        return x