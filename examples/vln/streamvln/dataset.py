# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Dataset for Visual Language Navigation

This dataset handles:
- 32-frame window segmentation
- Historical frame sampling (random/uniform)
- Multi-turn dialogue construction
- Action sequence prediction

Returns data in ms-swift standard format for multimodal training.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any
from PIL import Image
from torch.utils.data import Dataset

# Constants
DEFAULT_IMAGE_TOKEN = "<image>"


class StreamVLNDataset(Dataset):
    """
    StreamVLN Dataset for Visual Language Navigation training.
    
    Returns data in ms-swift standard format:
    {
        'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}],
        'images': [PIL.Image, ...]
    }
    
    Args:
        data_path: Path(s) to VLN data directory containing annotations.json.
                   Supports multiple paths separated by comma.
        num_frames: Window size (default: 32)
        num_history: Number of historical frames to sample (default: 8)
        num_future_steps: Number of actions per prediction round (default: 4)
        use_random: Use random sampling for historical frames (default: False for uniform)
        max_samples: Limit number of training samples
    """
    
    def __init__(
        self,
        data_path: str,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        use_random: bool = False,
        max_samples: Optional[int] = None,
    ):
        super(StreamVLNDataset, self).__init__()
        
        # VLN parameters
        self.num_frames = num_frames
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.use_random = use_random
        self.max_samples = max_samples
        
        # Load navigation data from multiple paths (comma-separated)
        self.video_folders = [p.strip() for p in data_path.split(',') if p.strip()]
        self.nav_data = []
        for vf in self.video_folders:
            anno_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_path):
                print(f"Warning: {anno_path} not found, skipping...")
                continue
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
            print(f"Loaded {len(anno_json)} episodes from {vf}")
        
        # Build data index: (episode_id, instruction_id, start_frame)
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item.get('instructions', [])
            actions = item.get('actions', [])
            actions_len = len(actions)
            
            if actions_len < 4:
                continue
            
            if not isinstance(instructions, list):
                instructions = [instructions]
            
            for ins_id in range(len(instructions)):
                # Segment trajectory into num_frames windows
                num_rounds = actions_len // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames))
        
        # Limit samples if max_samples is specified
        if self.max_samples is None:
            env_max_samples = os.getenv('VLN_MAX_SAMPLES')
            if env_max_samples is not None:
                try:
                    self.max_samples = int(env_max_samples)
                except ValueError:
                    self.max_samples = None
        
        if self.max_samples is not None and self.max_samples > 0:
            original_len = len(self.data_list)
            self.data_list = self.data_list[:self.max_samples]
            print(f"Limited dataset from {original_len} to {len(self.data_list)} samples (max_samples={self.max_samples})")
        
        # Action vocabulary
        self.idx2actions = {
            0: 'STOP',
            1: "↑",  # MOVE_FORWARD
            2: "←",  # TURN_LEFT
            3: "→",  # TURN_RIGHT
        }
        
        # Prompt templates
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        
        print(f"StreamVLNDataset initialized: {len(self.data_list)} samples from {len(self.nav_data)} episodes")
    
    def __len__(self):
        return len(self.data_list)
    
    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to text symbols."""
        if len(actions) == 0:
            return "STOP"
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(int(action), "STOP")
            converted_sequence.append(act_text)
        return ''.join(converted_sequence)
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get a training sample in ms-swift standard format.
        
        Returns:
            dict with keys:
                - messages: List of conversation turns
                - images: List of PIL Images
        """
        # DEBUG: 只在第一个样本时打印详细信息
        debug_print = (i == 0)
        
        if debug_print:
            print("\n" + "="*60)
            print(f"[DEBUG 阶段4.1] StreamVLNDataset.__getitem__({i}) 被调用")
            print("="*60)
        
        # Get sample index
        ep_id, ins_id, start_idx = self.data_list[i]
        
        if debug_print:
            print(f"[DEBUG] 样本索引: ep_id={ep_id}, ins_id={ins_id}, start_idx={start_idx}")
        data = self.nav_data[ep_id]
        
        # Get video frames
        video_path = data['video']
        rgb_path = os.path.join(video_path, 'rgb')
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB frames not found: {rgb_path}")
        video_frames = sorted(os.listdir(rgb_path))
        num_video_frames = len(video_frames)
        
        if num_video_frames == 0:
            raise ValueError(f"No frames found in: {rgb_path}")
        
        # Get instruction
        instructions = data.get("instructions", [])
        if not isinstance(instructions, list):
            instructions = [instructions]
        instruction = instructions[ins_id] if ins_id < len(instructions) else instructions[0]
        
        # Get actions (shift by 1 to predict next action)
        actions = data['actions'][1:] + [0]
        actions_len = len(actions)
        
        # Get time slice
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        if len(time_ids) == 0:
            time_ids = np.array([start_idx])
        current_actions = np.array(actions)[time_ids]
        
        # Sample current frames
        # Note: time_ids are indices in the shifted actions array, which directly correspond to video_frames array indices
        # video_frames[0] = 001.jpg corresponds to actions[0] (shifted), so no offset needed
        start_idx_abs = time_ids[0]
        end_idx_abs = time_ids[-1] + 1  # +1 because np.arange doesn't include end
        interval = self.num_future_steps
        
        sample_step_ids = np.arange(start_idx_abs, end_idx_abs, interval, dtype=np.int32)
        sample_step_ids = np.clip(sample_step_ids, 0, num_video_frames - 1)
        sample_step_ids = np.unique(sample_step_ids)
        
        if len(sample_step_ids) == 0:
            sample_step_ids = np.array([min(start_idx_abs, num_video_frames - 1)])
        
        sample_frame_paths = [os.path.join(rgb_path, video_frames[idx]) for idx in sample_step_ids]
        
        # Sample historical frames if not first segment
        history_frame_paths = []
        has_history = False
        if time_ids[0] != 0:
            current_start_abs = min(time_ids[0], num_video_frames)
            available_history_indices = np.arange(0, current_start_abs)
            num_to_sample = min(self.num_history, len(available_history_indices))
            
            if num_to_sample > 0:
                if self.use_random:
                    history_step_ids = np.random.choice(
                        available_history_indices,
                        size=num_to_sample,
                        replace=False
                    )
                    history_step_ids = np.sort(history_step_ids)
                else:
                    step = max(current_start_abs // self.num_history, 1)
                    history_step_ids = np.arange(0, current_start_abs, step)
                    if len(history_step_ids) > self.num_history:
                        indices = np.linspace(0, len(history_step_ids) - 1, self.num_history, dtype=int)
                        history_step_ids = history_step_ids[indices]
                
                history_step_ids = np.clip(history_step_ids, 0, num_video_frames - 1)
                history_frame_paths = [os.path.join(rgb_path, video_frames[idx]) for idx in history_step_ids]
                has_history = len(history_frame_paths) > 0
        
        # Load images as PIL Images
        # Note: Image processor will handle resize automatically (smart_resize)
        all_frame_paths = history_frame_paths + sample_frame_paths
        images = []
        for image_file in all_frame_paths:
            try:
                image = Image.open(image_file).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image {image_file}: {e}")
                # Create a black placeholder image
                images.append(Image.new('RGB', (640, 480), color='black'))
        
        if len(images) == 0:
            raise ValueError(f"No images loaded for sample {i}")
        
        # Build conversation in ms-swift format
        # System prompt
        system_prompt = (
            f"You are an autonomous navigation assistant. Your task is to {instruction}. "
            f"Devise an action sequence to follow the instruction using the four actions: "
            f"TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            f"MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        
        # Add history description if has history
        if has_history:
            num_history_images = len(history_frame_paths)
            history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
            system_prompt += f" These are your historical observations: {history_tokens}."
        
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Build multi-turn dialogue
        current_actions_list = list(current_actions)
        num_current_images = len(sample_frame_paths)
        
        action_idx = 0
        image_idx = 0
        while action_idx < len(current_actions_list) and image_idx < num_current_images:
            # User turn with image
            conjunction = random.choice(self.conjunctions)
            user_content = f"{conjunction}{DEFAULT_IMAGE_TOKEN}."
            messages.append({'role': 'user', 'content': user_content})
            
            # Assistant turn with actions
            step_actions = current_actions_list[action_idx:action_idx + self.num_future_steps]
            if len(step_actions) == 0:
                step_actions = [0]  # STOP
            answer = self.actions2text(step_actions)
            messages.append({'role': 'assistant', 'content': answer})
            
            action_idx += len(step_actions)
            image_idx += 1
        
        # DEBUG: 打印返回值信息
        if debug_print:
            print(f"\n[DEBUG] __getitem__ 返回值:")
            print(f"[DEBUG]   messages 数量: {len(messages)}")
            for j, msg in enumerate(messages[:3]):  # 只打印前3条
                content_preview = msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
                print(f"[DEBUG]     [{j}] {msg['role']}: {content_preview}")
            if len(messages) > 3:
                print(f"[DEBUG]     ... 还有 {len(messages)-3} 条消息")
            print(f"[DEBUG]   images 数量: {len(images)}")
            if images:
                print(f"[DEBUG]   第一张图像尺寸: {images[0].size}")
            print(f"[DEBUG] 这个返回值会被 template.encode() 处理!")
            print("="*60 + "\n")
        
        return {
            'messages': messages,
            'images': images,
        }
