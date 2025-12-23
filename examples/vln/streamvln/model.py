# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Model based on Qwen2.5-VL

This module extends Qwen2.5-VL with StreamVLN-specific features:
- 2D spatial pooling for image feature compression
- Memory token support for historical frame encoding
- Multi-frame processing for visual language navigation
"""

import math
import torch
import torch.nn.functional as F
from math import ceil
from typing import List, Optional, Union, Tuple

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# Constants (matching dataset)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
MEMORY_TOKEN_INDEX = -201


class StreamVLNQwen25VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for StreamVLN model based on Qwen2.5-VL.
    
    Extends Qwen2_5_VLConfig with StreamVLN-specific parameters.
    """
    
    def __init__(
        self,
        num_history: int = 8,
        current_stride: int = 2,
        history_stride: int = 2,
        spatial_pool_mode: str = "average",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_history = num_history
        self.current_stride = current_stride
        self.history_stride = history_stride
        self.spatial_pool_mode = spatial_pool_mode


class StreamVLNQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    StreamVLN model for Visual Language Navigation based on Qwen2.5-VL.
    
    Extends Qwen2_5_VL with:
    - Spatial pooling for image feature compression
    - Historical frame encoding as memory tokens
    - Multi-frame batch processing
    """
    
    config_class = StreamVLNQwen25VLConfig
    
    def __init__(self, config: StreamVLNQwen25VLConfig):
        super().__init__(config)
        
        # StreamVLN-specific parameters
        self.num_history = getattr(config, 'num_history', 8)
        self.current_stride = getattr(config, 'current_stride', 2)
        self.history_stride = getattr(config, 'history_stride', 2)
        self.spatial_pool_mode = getattr(config, 'spatial_pool_mode', 'average')
    
    def get_2d_pool(self, image_feature: torch.Tensor, stride: int = 2) -> torch.Tensor:
        """
        Apply 2D spatial pooling to image features.
        
        Args:
            image_feature: (num_frames, num_tokens, hidden_dim)
            stride: pooling stride
            
        Returns:
            pooled_feature: (num_frames, reduced_tokens, hidden_dim)
        """
        # Get spatial dimensions
        num_tokens = image_feature.shape[1]
        height = width = int(math.sqrt(num_tokens))
        
        if height * width != num_tokens:
            raise ValueError(f"Image features must be square, got {num_tokens} tokens")
        
        num_frames, _, hidden_dim = image_feature.shape
        
        # Reshape to 2D grid: (num_frames, height, width, hidden_dim)
        image_feature = image_feature.view(num_frames, height, width, hidden_dim)
        # Permute to: (num_frames, hidden_dim, height, width)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        # Apply pooling
        if self.spatial_pool_mode == "average":
            pooled = F.avg_pool2d(image_feature, kernel_size=stride, stride=stride)
        elif self.spatial_pool_mode == "max":
            pooled = F.max_pool2d(image_feature, kernel_size=stride, stride=stride)
        elif self.spatial_pool_mode == "bilinear":
            h, w = image_feature.shape[2:]
            scaled_shape = [ceil(h / stride), ceil(w / stride)]
            pooled = F.interpolate(image_feature, size=scaled_shape, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unexpected spatial_pool_mode: {self.spatial_pool_mode}")
        
        # Permute back: (num_frames, height', width', hidden_dim)
        pooled = pooled.permute(0, 2, 3, 1).contiguous()
        # Flatten spatial dims: (num_frames, height'*width', hidden_dim)
        pooled = pooled.view(num_frames, -1, hidden_dim)
        
        return pooled
    
    def encode_images_with_history(
        self,
        images: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Encode images and separate into current frames and historical memory.
        
        Args:
            images: (batch_size, num_images, 3, H, W)
            time_ids: (batch_size, num_frames) frame indices for each batch
            
        Returns:
            image_features: List of (num_current_frames, num_tokens, hidden_dim) per batch
            memory_features: List of (num_history_tokens, hidden_dim) or None per batch
        """
        batch_size = images.shape[0]
        num_images = images.shape[1]
        
        # Flatten batch and images: (batch_size * num_images, 3, H, W)
        images_flat = images.view(-1, *images.shape[2:])
        
        # Encode images using visual encoder
        vision_outputs = self.visual.forward(images_flat)
        
        # Get image features: (batch_size * num_images, num_tokens, hidden_dim)
        image_features_flat = vision_outputs.last_hidden_state
        
        # Reshape back: (batch_size, num_images, num_tokens, hidden_dim)
        num_tokens = image_features_flat.shape[1]
        hidden_dim = image_features_flat.shape[2]
        image_features = image_features_flat.view(batch_size, num_images, num_tokens, hidden_dim)
        
        # Separate history and current frames
        image_features_list = []
        memory_features_list = []
        
        for b in range(batch_size):
            # Check if this batch has history
            has_history = False
            if time_ids is not None and time_ids[b][0] > 0:
                has_history = True
            
            if has_history and num_images > self.num_history:
                # Separate historical and current frames
                history_feats = image_features[b, :self.num_history]  # (num_history, num_tokens, hidden_dim)
                current_feats = image_features[b, self.num_history:]  # (num_current, num_tokens, hidden_dim)
                
                # Compress historical frames
                history_pooled = self.get_2d_pool(history_feats, stride=self.history_stride)
                # Flatten history: (num_history * reduced_tokens, hidden_dim)
                memory_feat = history_pooled.view(-1, hidden_dim).unsqueeze(0)  # (1, total_tokens, hidden_dim)
                memory_features_list.append(memory_feat)
                
                # Compress current frames
                current_pooled = self.get_2d_pool(current_feats, stride=self.current_stride)
                image_features_list.append(current_pooled)
            else:
                # No history, all frames are current
                current_feats = image_features[b]  # (num_images, num_tokens, hidden_dim)
                current_pooled = self.get_2d_pool(current_feats, stride=self.current_stride)
                image_features_list.append(current_pooled)
                memory_features_list.append(None)
        
        return image_features_list, memory_features_list
    
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Prepare multimodal inputs by replacing image and memory tokens with features.
        
        This method:
        1. Encodes images and separates history/current
        2. Finds <image> and <memory> token positions
        3. Replaces them with actual visual features
        4. Constructs final input embeddings
        """
        # If no images or already past first token, return as is
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # Encode images
        image_features_list, memory_features_list = self.encode_images_with_history(images, time_ids)
        
        # Handle default values
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # Remove padding
        input_ids_list = [cur_input_ids[cur_attention_mask] 
                          for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels_list = [cur_labels[cur_attention_mask] 
                       for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # Process each sample in batch
        new_input_embeds = []
        new_labels = []
        
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            # Find special token positions
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            memory_token_indices = torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + memory_token_indices)
            
            if len(special_token_indices) == 0:
                # No special tokens, use regular embedding
                cur_input_embeds = self.get_input_embeddings()(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels_list[batch_idx])
                continue
            
            # Get token types at each special position
            special_tokens = [cur_input_ids[idx] for idx in special_token_indices]
            
            # Split input_ids at special token positions
            special_token_indices_with_bounds = [-1] + special_token_indices + [cur_input_ids.shape[0]]
            
            # Get text embeddings for non-special tokens
            cur_labels = labels_list[batch_idx]
            text_segments = []
            label_segments = []
            
            for i in range(len(special_token_indices_with_bounds) - 1):
                start = special_token_indices_with_bounds[i] + 1
                end = special_token_indices_with_bounds[i + 1]
                if start < end:
                    text_segments.append(cur_input_ids[start:end])
                    label_segments.append(cur_labels[start:end])
                else:
                    text_segments.append(torch.tensor([], dtype=cur_input_ids.dtype, device=cur_input_ids.device))
                    label_segments.append(torch.tensor([], dtype=cur_labels.dtype, device=cur_labels.device))
            
            # Get embeddings for text segments
            text_embeds = [self.get_input_embeddings()(seg) if len(seg) > 0 else 
                          torch.empty(0, self.config.hidden_size, device=cur_input_ids.device, dtype=self.dtype)
                          for seg in text_segments]
            
            # Interleave text embeddings with image/memory features
            cur_input_embeds = []
            cur_labels_new = []
            
            image_idx = 0
            memory_idx = 0
            
            for i in range(len(special_tokens) + 1):
                # Add text segment
                cur_input_embeds.append(text_embeds[i])
                cur_labels_new.append(label_segments[i])
                
                # Add special token feature if not at end
                if i < len(special_tokens):
                    if special_tokens[i] == IMAGE_TOKEN_INDEX:
                        # Add image features
                        img_feats = image_features_list[batch_idx][image_idx]  # (num_tokens, hidden_dim)
                        cur_input_embeds.append(img_feats)
                        # Images don't contribute to loss
                        cur_labels_new.append(
                            torch.full((img_feats.shape[0],), IGNORE_INDEX, 
                                      device=cur_labels.device, dtype=cur_labels.dtype)
                        )
                        image_idx += 1
                    elif special_tokens[i] == MEMORY_TOKEN_INDEX:
                        # Add memory features
                        mem_feats = memory_features_list[batch_idx]
                        if mem_feats is not None:
                            mem_feats = mem_feats.squeeze(0)  # Remove batch dim
                            cur_input_embeds.append(mem_feats)
                            cur_labels_new.append(
                                torch.full((mem_feats.shape[0],), IGNORE_INDEX,
                                          device=cur_labels.device, dtype=cur_labels.dtype)
                            )
                        memory_idx += 1
            
            # Concatenate all embeddings
            cur_input_embeds = torch.cat([x for x in cur_input_embeds if x.shape[0] > 0], dim=0)
            cur_labels_new = torch.cat([x for x in cur_labels_new if x.shape[0] > 0], dim=0)
            
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(cur_labels_new)
        
        # Pad to same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        new_input_embeds_padded = torch.zeros(
            batch_size, max_len, self.config.hidden_size,
            dtype=new_input_embeds[0].dtype, device=new_input_embeds[0].device
        )
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX,
            dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask_new = torch.zeros(
            batch_size, max_len,
            dtype=torch.bool, device=new_input_embeds[0].device
        )
        position_ids_new = torch.zeros(
            batch_size, max_len,
            dtype=torch.long, device=new_input_embeds[0].device
        )
        
        for i, (embeds, labs) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = embeds.shape[0]
            new_input_embeds_padded[i, :cur_len] = embeds
            new_labels_padded[i, :cur_len] = labs
            attention_mask_new[i, :cur_len] = True
            position_ids_new[i, :cur_len] = torch.arange(cur_len, device=embeds.device)
        
        return None, position_ids_new, attention_mask_new, past_key_values, new_input_embeds_padded, new_labels_padded
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for StreamVLN model.
        
        Processes multimodal inputs with image and memory tokens.
        """
        if inputs_embeds is None and images is not None:
            # Prepare multimodal inputs
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                time_ids,
                **kwargs
            )
        
        # Call parent forward
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


# Register model for auto loading with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM
    AutoModel.register(StreamVLNQwen25VLConfig, StreamVLNQwen25VLForConditionalGeneration)
    AutoModelForCausalLM.register(StreamVLNQwen25VLConfig, StreamVLNQwen25VLForConditionalGeneration)
except Exception:
    pass


def get_model_tokenizer_streamvln_qwen2_5_vl(model_dir, *args, **kwargs):
    """
    Load StreamVLN Qwen2.5-VL model and processor.
    
    Uses Qwen2.5-VL's loading function as base, then loads StreamVLN model.
    """
    from transformers import AutoProcessor
    
    # Load processor (contains tokenizer)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    # Load model using Qwen2.5-VL base class
    model = StreamVLNQwen25VLForConditionalGeneration.from_pretrained(
        model_dir,
        trust_remote_code=True,
        **kwargs.get('model_kwargs', {})
    )
    
    return model, processor
