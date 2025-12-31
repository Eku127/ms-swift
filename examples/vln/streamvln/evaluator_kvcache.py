# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Evaluator with KV Cache for Habitat VLN tasks.

This evaluator implements INCREMENTAL inference using KV cache:
- First turn: Full encoding with system prompt + first user turn
- Subsequent turns: Only encode new user turn, use past_key_values for context
- Window transition: Reset KV cache and re-sample history

Key difference from evaluator.py:
- Uses KV cache for efficient incremental inference
- Does NOT re-encode the full conversation each step
- Manually handles token construction to avoid automatic system prompt insertion

Note: This approach is more efficient but requires careful handling of:
- Token boundaries (im_end, im_start markers)
- Image placeholder alignment
- KV cache state management
"""

import os
import re
import torch
import numpy as np
import random
import copy
from typing import Any, List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

import habitat
from habitat import Env
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

# Trigger registration of measures
try:
    from . import habitat_extensions
except ImportError:
    import habitat_extensions

# Constants matching dataset.py
DEFAULT_IMAGE_TOKEN = "<image>"


class VLNEvaluatorKVCache:
    """
    VLN Evaluator with KV Cache for efficient incremental inference.
    
    This evaluator uses past_key_values to avoid re-encoding the entire
    conversation at each step. Within a window:
    - First turn: Full encoding
    - Subsequent turns: Only new user turn, reusing KV cache
    """
    
    def __init__(
        self,
        config_path: str,
        model: Any,
        processor: Any,
        template: Any,
        args: Any,
    ):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.processor = processor
        self.template = template
        self.tokenizer = processor.tokenizer
        
        # Habitat config
        from habitat_baselines.config.default import get_config as get_habitat_config
        self.config = get_habitat_config(config_path)
        
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = args.eval_split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        # Action mapping
        self.idx2actions = {
            0: 'STOP',
            1: "↑",
            2: "←",
            3: "→",
        }
        self.actions2idx = {v: k for k, v in self.idx2actions.items()}
        
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
        
        # VLN parameters
        self.num_frames = args.num_frames
        self.num_history = args.num_history
        self.num_future_steps = args.num_future_steps
        
        # Video generation
        self.save_video = getattr(args, 'save_video', False)
        self.output_path = getattr(args, 'output_dir', './results/eval')
        
    def config_env(self) -> Env:
        return Env(config=self.config)

    def parse_actions(self, output: str) -> List[int]:
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        return actions

    def sample_history_indices(self, total_frames: int, num_to_sample: int) -> List[int]:
        if total_frames <= 0:
            return []
        if total_frames <= num_to_sample:
            return list(range(total_frames))
        step = max(total_frames // num_to_sample, 1)
        indices = list(range(0, total_frames, step))
        if len(indices) > num_to_sample:
            selected = np.linspace(0, len(indices) - 1, num_to_sample, dtype=int)
            indices = [indices[i] for i in selected]
        return indices

    def build_system_prompt(self, instruction: str, num_history_images: int = 0) -> str:
        system_prompt = (
            f"You are an autonomous navigation assistant. Your task is to {instruction}. "
            f"Devise an action sequence to follow the instruction using the four actions: "
            f"TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            f"MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        if num_history_images > 0:
            history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
            system_prompt += f" These are your historical observations: {history_tokens}."
        return system_prompt

    def build_first_turn_messages(
        self,
        instruction: str,
        rgb_list: List[Image.Image],
        window_start: int,
        current_image: Image.Image,
    ) -> Tuple[List[Dict], List[Image.Image]]:
        """Build complete messages for the FIRST turn in a window."""
        history_images = []
        if window_start > 0:
            history_indices = self.sample_history_indices(window_start, self.num_history)
            history_images = [rgb_list[i] for i in history_indices]
        
        system_prompt = self.build_system_prompt(
            instruction,
            num_history_images=len(history_images)
        )
        
        conjunction = random.choice(self.conjunctions)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"{conjunction}{DEFAULT_IMAGE_TOKEN}."}
        ]
        
        images = history_images + [current_image]
        return messages, images, conjunction

    def encode_subsequent_turn(
        self,
        current_image: Image.Image,
        previous_response: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode ONLY the subsequent turn for incremental inference.
        
        This constructs:
        <|im_end|>
        <|im_start|>assistant
        {previous_response}<|im_end|>
        <|im_start|>user
        {conjunction}<image>.<|im_end|>
        <|im_start|>assistant
        
        And processes the image separately.
        """
        conjunction = random.choice(self.conjunctions)
        
        # Manually construct the token sequence for subsequent turn
        # This avoids the automatic system prompt insertion
        subsequent_text = (
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{previous_response}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{conjunction}{DEFAULT_IMAGE_TOKEN}.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # Use template to encode just this user turn with image
        # We create a minimal messages structure
        messages = [
            {'role': 'user', 'content': f"{conjunction}{DEFAULT_IMAGE_TOKEN}."}
        ]
        
        encoded = self.template.encode({'messages': messages, 'images': [current_image]})
        
        # Find and remove the automatic system prompt from encoded input_ids
        # The system prompt typically looks like:
        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        input_ids = encoded['input_ids']
        
        # Find the position after the system prompt ends
        # Look for the pattern: <|im_start|>user after <|im_end|>
        im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        
        # Find first <|im_end|> (end of system) then find <|im_start|> (start of user)
        end_of_system = None
        for i, token_id in enumerate(input_ids):
            if token_id == im_end_id:
                # Check if next tokens indicate user turn
                if i + 1 < len(input_ids):
                    end_of_system = i + 1  # Position after <|im_end|>
                    # Skip the newline if present
                    while end_of_system < len(input_ids) and input_ids[end_of_system] == self.tokenizer.convert_tokens_to_ids('\n'):
                        end_of_system += 1
                    break
        
        if end_of_system is not None:
            # Remove system prompt, keep only user turn part
            input_ids = input_ids[end_of_system:]
        
        # Prepend the previous assistant response with proper formatting
        # <|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n
        prefix_text = f"<|im_end|>\n<|im_start|>assistant\n{previous_response}<|im_end|>\n"
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        
        # Combine: prefix + user turn
        new_input_ids = prefix_ids + input_ids
        
        return torch.tensor([new_input_ids]), encoded, conjunction

    def append_text_to_image(self, image: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
        """Append text to image."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        font_size = 20
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(pil_image)
        max_width = pil_image.width - 20
        words = text.split(' ')
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            word_width = bbox[2] - bbox[0]
            if current_width + word_width <= max_width or len(current_line) == 0:
                current_line.append(word)
                current_width = word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        if current_line:
            lines.append(' '.join(current_line))
        
        line_height = font_size + 4
        text_height = len(lines) * line_height
        padding = 10
        new_height = pil_image.height + text_height + 2 * padding
        new_image = Image.new('RGB', (pil_image.width, new_height), color='white')
        
        if position == 'top':
            draw = ImageDraw.Draw(new_image)
            y_pos = padding
            for line in lines:
                draw.text((padding, y_pos), line, fill='black', font=font)
                y_pos += line_height
            new_image.paste(pil_image, (0, text_height + 2 * padding))
        else:
            new_image.paste(pil_image, (0, 0))
            draw = ImageDraw.Draw(new_image)
            y_pos = pil_image.height + padding
            for line in lines:
                draw.text((padding, y_pos), line, fill='black', font=font)
                y_pos += line_height
        
        return np.array(new_image)

    @torch.no_grad()
    def eval_episode(self, env: Env, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single episode using KV cache incremental inference.
        
        Key difference from VLNEvaluator:
        - First turn: Full encoding
        - Subsequent turns: Only encode new turn, use past_key_values
        """
        self.model.eval()
        self.model.reset_for_env(env_idx)
        
        env.current_episode = episode
        observations = env.reset()
        
        instruction = episode.instruction.instruction_text
        episode_id = episode.episode_id
        scene_id = episode.scene_id.split('/')[-2] if '/' in episode.scene_id else episode.scene_id
        
        # State tracking
        rgb_list = []
        action_seq = []
        step_id = 0
        
        # Video generation
        vis_frames = []
        if self.save_video:
            video_dir = os.path.join(self.output_path, 'videos', f'{scene_id}_{episode_id}')
            os.makedirs(video_dir, exist_ok=True)
        
        # KV cache state
        past_key_values = None
        last_response = None
        window_start = 0
        
        max_steps = self.config.habitat.environment.max_episode_steps
        
        while not env.episode_over and step_id < max_steps:
            rgb = observations["rgb"]
            current_img = Image.fromarray(rgb).convert('RGB')
            rgb_list.append(current_img)
            
            if len(action_seq) == 0:
                # Check window transition
                if step_id % self.num_frames == 0:
                    window_start = step_id
                    past_key_values = None
                    last_response = None
                
                try:
                    if past_key_values is None:
                        # === FIRST TURN IN WINDOW ===
                        messages, images, _ = self.build_first_turn_messages(
                            instruction=instruction,
                            rgb_list=rgb_list,
                            window_start=window_start,
                            current_image=current_img,
                        )
                        
                        encoded = self.template.encode({'messages': messages, 'images': images})
                        input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                        
                        model_inputs = {'input_ids': input_ids}
                        
                        if 'pixel_values' in encoded:
                            pv = encoded['pixel_values']
                            model_inputs['pixel_values'] = pv.to(self.device).to(self.model.dtype) if isinstance(pv, torch.Tensor) else torch.tensor(pv).to(self.device).to(self.model.dtype)
                        
                        if 'image_grid_thw' in encoded:
                            gt = encoded['image_grid_thw']
                            model_inputs['image_grid_thw'] = gt.to(self.device) if isinstance(gt, torch.Tensor) else torch.tensor(gt).to(self.device)
                        
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=64,
                            do_sample=False,
                            use_cache=True,
                            return_dict_in_generate=True,
                        )
                        
                        past_key_values = outputs.past_key_values
                        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
                        
                    else:
                        # === SUBSEQUENT TURN IN WINDOW ===
                        new_input_ids, encoded, _ = self.encode_subsequent_turn(
                            current_image=current_img,
                            previous_response=last_response,
                        )
                        new_input_ids = new_input_ids.to(self.device)
                        
                        model_inputs = {'input_ids': new_input_ids}
                        
                        if 'pixel_values' in encoded:
                            pv = encoded['pixel_values']
                            model_inputs['pixel_values'] = pv.to(self.device).to(self.model.dtype) if isinstance(pv, torch.Tensor) else torch.tensor(pv).to(self.device).to(self.model.dtype)
                        
                        if 'image_grid_thw' in encoded:
                            gt = encoded['image_grid_thw']
                            model_inputs['image_grid_thw'] = gt.to(self.device) if isinstance(gt, torch.Tensor) else torch.tensor(gt).to(self.device)
                        
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=64,
                            do_sample=False,
                            use_cache=True,
                            return_dict_in_generate=True,
                            past_key_values=past_key_values,
                        )
                        
                        past_key_values = outputs.past_key_values
                        generated_ids = outputs.sequences[0][new_input_ids.shape[1]:]
                    
                    output_text = self.processor.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    ).strip()
                    
                    last_response = output_text
                    action_seq = self.parse_actions(output_text)
                    
                    if getattr(self.args, 'verbose', False):
                        print(f"Step {step_id}: {output_text} -> {action_seq}")
                    
                except Exception as e:
                    print(f"[Warning] Generation failed at step {step_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    action_seq = []
                
                if not action_seq:
                    action_seq = [0]
            
            # Collect vis frame
            if self.save_video:
                info = env.get_metrics()
                if info.get('top_down_map') is not None:
                    frame = observations_to_image({'rgb': observations['rgb']}, info)
                    frame = self.append_text_to_image(frame, f"Instruction: {instruction}", position='top')
                    vis_frames.append(frame)
            
            action = action_seq.pop(0)
            observations = env.step(action)
            step_id += 1
        
        metrics = env.get_metrics()
        
        if self.save_video and len(vis_frames) > 0:
            try:
                images_to_video(
                    vis_frames, 
                    os.path.join(self.output_path, 'videos'), 
                    f'{scene_id}_{episode_id}',
                    fps=6,
                    quality=9
                )
            except Exception as e:
                print(f"[Warning] Failed to save video: {e}")
        
        return metrics


