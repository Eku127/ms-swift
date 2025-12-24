# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Module for ms-swift

This module provides VLN (Visual Language Navigation) training support using
Qwen2.5-VL as the base model. It leverages ms-swift's native multimodal
processing capabilities.

Components:
    1. Model Registration - Registers StreamVLN model type with ms-swift
    2. Dataset - Custom VLN dataset with trajectory segmentation
    3. Training Arguments - VLN-specific parameters
    4. Trainer - Custom SFT trainer for VLN

Usage:
    python examples/vln/streamvln/trainer.py \\
        --custom_register_path examples/vln/streamvln \\
        --model_type streamvln_qwen2_5_vl \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --dataset /path/to/vln_data \\
        ...

Architecture:
    The model uses Qwen2.5-VL's native template for multimodal processing:
    
    Dataset returns: {'messages': [...], 'images': [PIL.Image, ...]}
                              ↓
    Template.encode() processes images and text
                              ↓
    Model receives pre-fused inputs_embeds
                              ↓
    Standard LM forward pass
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
register_model(
    ModelMeta(
        model_type='streamvln_qwen2_5_vl',
        model_groups=[
            ModelGroup([
                Model('streamvln-qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('streamvln-qwen2.5-vl-7b', 'Qwen/Qwen2.5-VL-7B-Instruct'),
            ])
        ],
        template=TemplateType.qwen2_5_vl,  # Use native Qwen2.5-VL template
        get_function=get_model_tokenizer_streamvln_qwen2_5_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['StreamVLNQwen25VLForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6'],
        tags=['vision', 'vln', 'navigation'],
        is_multimodal=True,
    )
)

print("[StreamVLN] Model 'streamvln_qwen2_5_vl' registered successfully!")


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
