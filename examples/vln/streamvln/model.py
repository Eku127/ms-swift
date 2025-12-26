# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Model based on Qwen2.5-VL

This module provides a StreamVLN model that inherits from Qwen2.5-VL.
The model uses Qwen2.5-VL's native multimodal processing capabilities,
with the ms-swift framework handling all image preprocessing and fusion.

Architecture:
    StreamVLNQwen25VLForConditionalGeneration
            ↓ inherits
    Qwen2_5_VLForConditionalGeneration (from transformers)

The ms-swift framework processes images through Qwen2.5-VL's template,
which handles:
    1. Image preprocessing → pixel_values
    2. Vision encoding → image features  
    3. Multimodal fusion → inputs_embeds
    4. Forward pass with fused embeddings
"""

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig


class StreamVLNQwen25VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for StreamVLN model based on Qwen2.5-VL.
    
    Inherits all configuration from Qwen2_5_VLConfig.
    Can be extended with StreamVLN-specific parameters if needed.
    """
    model_type = "streamvln_qwen2_5_vl"


class StreamVLNQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    StreamVLN model for Visual Language Navigation based on Qwen2.5-VL.
    
    This model directly inherits from Qwen2_5_VLForConditionalGeneration without
    overriding any methods. The ms-swift framework handles all multimodal processing
    through the Qwen2.5-VL template.
    
    Key points:
    - All image processing is handled by the framework (pixel_values, image_grid_thw)
    - Multimodal fusion (text + image) is done before calling forward()
    - The model receives pre-fused inputs_embeds, not raw images
    
    Usage:
        This model is registered with ms-swift via register_model() in __init__.py.
        It uses the Qwen2.5-VL template (TemplateType.qwen2_5_vl) for processing.
    """
    
    config_class = StreamVLNQwen25VLConfig


# Register model for auto loading with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
    AutoConfig.register("streamvln_qwen2_5_vl", StreamVLNQwen25VLConfig)
    AutoModel.register(StreamVLNQwen25VLConfig, StreamVLNQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(StreamVLNQwen25VLConfig, StreamVLNQwen25VLForConditionalGeneration)
except Exception:
    pass


def get_model_tokenizer_streamvln_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    """
    Load StreamVLN Qwen2.5-VL model and processor.
    
    This function is called by ms-swift when loading the registered model.
    It uses the standard Qwen2.5-VL loading function to properly handle
    attn_impl, torch_dtype, and other parameters.
    
    Args:
        model_dir: Path to the model directory (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')
        model_info: ModelInfo object from ms-swift
        model_kwargs: Additional kwargs for model loading
        load_model: Whether to load the model weights
        **kwargs: Additional arguments including attn_impl, torch_dtype, etc.
        
    Returns:
        tuple: (model, processor)
    """
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_vl
    
    # Use the standard Qwen2.5-VL loader which properly handles:
    # - attn_impl (flash_attn, sdpa, eager)
    # - torch_dtype
    # - automodel_class (Qwen2_5_VLForConditionalGeneration)
    # - qwen_vl_utils compatibility
    return get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
