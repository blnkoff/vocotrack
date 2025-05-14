from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Any, Iterable


class BlockType(StrEnum):
    BASIC = 'basic'
    BOTTLENECK = 'bottleneck'


class ConvCfg(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int] | int
    stride: int
    padding: int | None = None


def _stem_default(values: dict[str, Any]) -> ConvCfg:
    if values['block_type'] is BlockType.BASIC:
        out_channels = 32
    else:
        out_channels = 64

    return ConvCfg(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=3,
        stride=1
    )
    
def _dropout_prob(values: dict[str, Any]) -> float:
    return values['resnet'].dropout_prob

class StageCfg(BaseModel):
    in_channels: int
    num_channels: int
    blocks: int
    stride: int


class ResNetCfg(BaseModel):
    block_type: BlockType = BlockType.BASIC
    stem: ConvCfg = Field(default_factory=_stem_default)
    stages: Iterable[StageCfg]
    dropout_prob: float = 0.2


class RVectorCfg(BaseModel):
    resnet: ResNetCfg
    emb_gender: int = 16
    emb_word: int = 64
    dropout_prob: float = Field(default_factory=_dropout_prob)
