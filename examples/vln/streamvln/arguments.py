# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Training Arguments

Defines custom training parameters for StreamVLN.
"""

from dataclasses import dataclass, field
from typing import Optional

from swift.llm import TrainArguments


@dataclass
class StreamVLNTrainArguments(TrainArguments):
    """
    StreamVLN 专属训练参数
    
    继承 ms-swift TrainArguments，添加 VLN 特有的参数。
    """
    # VLN arguments
    num_frames: int = field(
        default=32, 
        metadata={"help": "Window size for trajectory segmentation"}
    )
    num_history: int = field(
        default=8, 
        metadata={"help": "Number of historical frames to sample"}
    )
    num_future_steps: int = field(
        default=4, 
        metadata={"help": "Number of actions to predict per round"}
    )
    use_random: bool = field(
        default=False, 
        metadata={"help": "Use random sampling for history (false=uniform)"}
    )
    
    # Limit dataset size
    vln_max_samples: Optional[int] = field(
        default=None, 
        metadata={"help": "Limit number of training samples"}
    )
