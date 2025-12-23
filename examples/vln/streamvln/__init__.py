# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Module for ms-swift

This module provides:
1. Custom model registration (StreamVLN based on Qwen2.5-VL)
2. Custom dataset registration
3. Custom training arguments and trainer

Usage:
    python examples/vln/streamvln/trainer.py \
        --custom_register_path examples/vln/streamvln \
        --model_type streamvln_qwen2_5_vl \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --dataset /path/to/vln_data \
        ...
"""

# =============================================================================
# Model Registration
# =============================================================================

from swift.llm import register_model, ModelMeta, ModelGroup, Model
from swift.llm.template import TemplateType
from swift.llm.model import ModelArch

from .model import (
    StreamVLNQwen25VLConfig,
    StreamVLNQwen25VLForConditionalGeneration,
    get_model_tokenizer_streamvln_qwen2_5_vl,
)

# Register StreamVLN model based on Qwen2.5-VL
# Using string for model_type to avoid modifying constant.py
register_model(
    ModelMeta(
        model_type='streamvln_qwen2_5_vl',  # String, not enum
        model_groups=[
            ModelGroup([
                Model('streamvln-qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('streamvln-qwen2.5-vl-7b', 'Qwen/Qwen2.5-VL-7B-Instruct'),
            ])
        ],
        template=TemplateType.qwen2_5_vl,
        get_function=get_model_tokenizer_streamvln_qwen2_5_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['StreamVLNQwen25VLForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6'],
        tags=['vision', 'vln', 'navigation'],
        is_multimodal=True,  # 必须显式设置，因为自定义 model_type 不在 MLLMModelType 中
    )
)

print("[StreamVLN] Custom model 'streamvln_qwen2_5_vl' registered successfully!")


# =============================================================================
# Dataset Registration (Optional - for compatibility)
# =============================================================================

from swift.llm import register_dataset, DatasetMeta
from datasets import Dataset as HfDataset
from typing import Optional
import os


def streamvln_load_function(
    dataset_syntax,
    dataset_meta: Optional[DatasetMeta] = None,
    **kwargs
) -> HfDataset:
    """
    Custom load function for StreamVLN dataset.
    
    This function detects StreamVLN dataset paths (containing annotations.json)
    and returns a placeholder HuggingFace Dataset that will be replaced
    with the actual StreamVLNDataset in the training code.
    
    If the path is not a StreamVLN dataset, falls back to default loader.
    """
    import sys
    print(f"[StreamVLN] streamvln_load_function called with path: {dataset_syntax.dataset}", 
          file=sys.stderr, flush=True)
    
    # Get data path from dataset_syntax
    data_path = dataset_syntax.dataset
    
    # Check if this is a StreamVLN dataset path
    if os.path.isdir(data_path):
        annotations_path = os.path.join(data_path, 'annotations.json')
        if os.path.exists(annotations_path):
            # Create a placeholder HuggingFace Dataset with metadata
            dataset = HfDataset.from_dict({
                '_streamvln_marker': [True],
            })
            
            # Store metadata as attributes for trainer to access
            dataset._streamvln_data_path = data_path
            dataset._is_streamvln = True
            
            print(f"[StreamVLN] Detected StreamVLN dataset at: {data_path}", 
                  file=sys.stderr, flush=True)
            return dataset
    
    # Not a StreamVLN dataset, fallback to default loader
    print(f"[StreamVLN] Path is not a StreamVLN dataset, falling back to default loader", 
          file=sys.stderr, flush=True)
    from swift.llm.dataset.loader import DatasetLoader
    return DatasetLoader.load(dataset_syntax, dataset_meta, **kwargs)


# Register StreamVLN dataset loader
register_dataset(
    DatasetMeta(
        dataset_path=None,  # Will be matched via load_function logic
        load_function=streamvln_load_function,
        split=['train'],
        tags=['multi-modal', 'vision', 'vln', 'navigation'],
        huge_dataset=False,
    )
)

print("[StreamVLN] Custom dataset registration loaded successfully!")


# =============================================================================
# Public API
# =============================================================================

from .dataset import StreamVLNDataset
from .arguments import StreamVLNTrainArguments

__all__ = [
    # Model
    'StreamVLNQwen25VLConfig',
    'StreamVLNQwen25VLForConditionalGeneration',
    'get_model_tokenizer_streamvln_qwen2_5_vl',
    # Dataset
    'StreamVLNDataset',
    # Training
    'StreamVLNTrainArguments',
]

# Note: StreamVLNSft and train_main are imported from trainer.py directly
# to avoid circular imports when running trainer.py as a script
