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


def get_model_tokenizer_streamvln_qwen2_5_vl(model_dir, *args, **kwargs):
    """
    Load StreamVLN Qwen2.5-VL model and processor.
    
    This function is called by ms-swift when loading the registered model.
    It loads the base Qwen2.5-VL model with our StreamVLN config/class.
    
    Args:
        model_dir: Path to the model directory (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')
        *args, **kwargs: Additional arguments passed by ms-swift
        
    Returns:
        tuple: (model, processor)
    """
    from transformers import AutoProcessor
    
    # Load processor (contains tokenizer and image processor)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    # Load model - use base Qwen2.5-VL since we don't override forward()
    # The StreamVLN wrapper is mainly for registration and future extensions
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        **kwargs.get('model_kwargs', {})
    )
    
    return model, processor
