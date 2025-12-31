# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Evaluator for Habitat VLN tasks.

This evaluator implements streaming inference for Visual Language Navigation,
adapted for ms-swift's StreamVLNQwen25VL model (RGB-only, multiple <image> tokens).

Complete Multi-turn Dialogue Inference:
=======================================

Each inference call builds COMPLETE messages matching training format:
1. System prompt (with historical <image> tokens if applicable)
2. All previous user/assistant turns within the current window
3. Current user turn (waiting for model response)

This approach ensures:
- No automatic system prompt insertion (we explicitly provide it)
- Image count matches <image> placeholder count exactly
- Format matches training data (dataset.py) exactly

Window Management:
- Every num_frames steps, reset window state and re-sample history
- Within window, accumulate dialogue turns with cached responses

Prompt Format (matching dataset.py):
====================================
System: "You are an autonomous navigation assistant. Your task is to {instruction}. 
         Devise an action sequence... These are your historical observations: <image>..."
User: "{conjunction}<image>."  (e.g., "you can see <image>.")
Assistant: "{action_sequence}"  (e.g., "↑↑←→" or "STOP")
User: "{conjunction}<image>."
Assistant: "{action_sequence}"
...
"""

import os
import re
import torch
import numpy as np
import random
import copy
import contextlib
import io
import warnings
import logging
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
# Handle both module and direct import cases
try:
    from . import habitat_extensions
except ImportError:
    import habitat_extensions

# Constants matching dataset.py
DEFAULT_IMAGE_TOKEN = "<image>"


class VLNEvaluator:
    """
    VLN Evaluator implementing streaming inference for StreamVLNQwen25VL.
    
    Key features:
    - RGB-only input (no depth/gps/compass)
    - Historical frames represented by multiple <image> tokens
    - Window-based inference with num_frames window size
    - Dialogue format consistent with training (dataset.py)
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
        
        # Habitat config
        from habitat_baselines.config.default import get_config as get_habitat_config
        self.config = get_habitat_config(config_path)
        
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = args.eval_split
            # Add metrics
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

        # Action mapping (same as dataset.py)
        self.idx2actions = {
            0: 'STOP',
            1: "↑",  # MOVE_FORWARD
            2: "←",  # TURN_LEFT
            3: "→",  # TURN_RIGHT
        }
        self.actions2idx = {v: k for k, v in self.idx2actions.items()}
        
        # Prompt templates (same as dataset.py)
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
        
        # Video generation parameters
        self.save_video = getattr(args, 'save_video', False)
        self.output_path = getattr(args, 'output_dir', './results/eval')
        
    def config_env(self) -> Env:
        """Create and return a Habitat environment."""
        return Env(config=self.config)
    
    def append_text_to_image(self, image: np.ndarray, text: str, position: str = 'bottom') -> np.ndarray:
        """
        Append text to image (below the image).
        
        Args:
            image: Input image as numpy array (H, W, C)
            text: Text to add
            position: 'bottom' or 'top'
            
        Returns:
            Image with text appended below
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Try to load a font, fallback to default if not available
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
        
        # Calculate text size (handle text wrapping for long instructions)
        draw = ImageDraw.Draw(pil_image)
        max_width = pil_image.width - 20  # Leave margins
        
        # Simple text wrapping
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
        
        # Calculate total text height
        line_height = font_size + 4
        text_height = len(lines) * line_height
        
        # Create new image with space for text
        padding = 10
        new_height = pil_image.height + text_height + 2 * padding
        new_image = Image.new('RGB', (pil_image.width, new_height), color='white')
        
        if position == 'bottom':
            # Paste original image at top
            new_image.paste(pil_image, (0, 0))
            # Draw text at bottom
            draw = ImageDraw.Draw(new_image)
            y_pos = pil_image.height + padding
            for line in lines:
                draw.text((padding, y_pos), line, fill='black', font=font)
                y_pos += line_height
        else:
            # Draw text at top
            draw = ImageDraw.Draw(new_image)
            y_pos = padding
            for line in lines:
                draw.text((padding, y_pos), line, fill='black', font=font)
                y_pos += line_height
            # Paste original image below text
            new_image.paste(pil_image, (0, text_height + 2 * padding))
        
        return np.array(new_image)

    def parse_actions(self, output: str) -> List[int]:
        """
        Parse action sequence from model output string.
        
        Args:
            output: Model output text containing action symbols (↑←→STOP)
            
        Returns:
            List of action indices
        """
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        return actions

    def sample_history_indices(self, total_frames: int, num_to_sample: int) -> List[int]:
        """
        Uniformly sample indices from historical frames.
        
        This matches the original StreamVLN's history sampling strategy and
        the training dataset's sampling logic.
        
        Args:
            total_frames: Total number of historical frames available
            num_to_sample: Number of frames to sample
            
        Returns:
            List of sampled frame indices
        """
        if total_frames <= 0:
            return []
        
        if total_frames <= num_to_sample:
            return list(range(total_frames))
        
        # Uniform sampling like original StreamVLN
        step = max(total_frames // num_to_sample, 1)
        indices = list(range(0, total_frames, step))
        
        # Ensure we don't exceed num_to_sample
        if len(indices) > num_to_sample:
            selected = np.linspace(0, len(indices) - 1, num_to_sample, dtype=int)
            indices = [indices[i] for i in selected]
        
        return indices

    def build_system_prompt(self, instruction: str, num_history_images: int = 0) -> str:
        """
        Build system prompt consistent with training (dataset.py).
        
        Format:
        "You are an autonomous navigation assistant. Your task is to {instruction}. 
        Devise an action sequence to follow the instruction using the four actions: 
        TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, 
        MOVE FORWARD (↑) by 25 centimeters, or STOP."
        
        If has history:
        " These are your historical observations: <image> <image> ..."
        
        Args:
            instruction: Navigation instruction text
            num_history_images: Number of historical observation images
            
        Returns:
            System prompt string
        """
        system_prompt = (
            f"You are an autonomous navigation assistant. Your task is to {instruction}. "
            f"Devise an action sequence to follow the instruction using the four actions: "
            f"TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            f"MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        
        # Add history description if has history (matching dataset.py format)
        if num_history_images > 0:
            history_tokens = ' '.join([DEFAULT_IMAGE_TOKEN for _ in range(num_history_images)])
            system_prompt += f" These are your historical observations: {history_tokens}."
        
        return system_prompt

    def build_complete_messages(
        self,
        instruction: str,
        rgb_list: List[Image.Image],
        window_start: int,
        current_step: int,
        window_responses: List[str],
        window_conjunctions: List[str],
    ) -> Tuple[List[Dict], List[Image.Image]]:
        """
        Build COMPLETE multi-turn messages for inference (matching training format).
        
        This method constructs the full dialogue including:
        - System prompt (with historical <image> tokens if applicable)
        - All user/assistant turns within the current window
        - Current user turn (waiting for assistant response)
        
        This approach ensures:
        1. No automatic system prompt insertion (we explicitly provide it)
        2. Image count matches <image> placeholder count
        3. Format matches training data exactly
        
        Args:
            instruction: Navigation instruction
            rgb_list: All collected RGB images
            window_start: Start index of current window
            current_step: Current step index
            window_responses: Cached assistant responses for previous turns in this window
            window_conjunctions: Cached conjunctions used for previous turns
            
        Returns:
            Tuple of (messages, images)
        """
        # 1. Sample historical frames (from before current window)
        history_images = []
        if window_start > 0:
            history_indices = self.sample_history_indices(window_start, self.num_history)
            history_images = [rgb_list[i] for i in history_indices]
        
        # 2. Build system prompt with history
        system_prompt = self.build_system_prompt(
            instruction,
            num_history_images=len(history_images)
        )
        
        # 3. Start with system message
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # 4. Build multi-turn dialogue for current window
        # This matches dataset.py's structure exactly:
        # - One image every num_future_steps actions
        # - Each image corresponds to num_future_steps actions
        window_images = []
        
        # Calculate number of image turns: each turn covers num_future_steps actions
        # For example, with num_future_steps=4:
        # - step 0-3: use image at step 0, turn 0
        # - step 4-7: use image at step 4, turn 1
        # - etc.
        steps_in_window = current_step - window_start + 1
        num_turns = (steps_in_window - 1) // self.num_future_steps + 1
        
        for i in range(num_turns):
            # Image is sampled at the start of each turn (every num_future_steps)
            step_idx = window_start + i * self.num_future_steps
            
            # Get or generate conjunction for this turn
            if i < len(window_conjunctions):
                conjunction = window_conjunctions[i]
            else:
                conjunction = random.choice(self.conjunctions)
            
            # User turn with image
            messages.append({'role': 'user', 'content': f"{conjunction}{DEFAULT_IMAGE_TOKEN}."})
            window_images.append(rgb_list[step_idx])
            
            # Assistant turn (except for the last turn which we're generating)
            if i < num_turns - 1:
                # Use cached response from previous inference
                if i < len(window_responses):
                    messages.append({'role': 'assistant', 'content': window_responses[i]})
                else:
                    # Fallback (shouldn't happen if logic is correct)
                    messages.append({'role': 'assistant', 'content': '↑'})
        
        # 5. Combine images: history + window
        images = history_images + window_images
        
        return messages, images

    @torch.no_grad()
    def eval_episode(self, env: Env, episode: Any, env_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single episode using complete multi-turn dialogue inference.
        
        This approach builds COMPLETE messages for each inference call, matching
        the training data format exactly:
        
        1. System prompt (with historical <image> tokens if applicable)
        2. All previous user/assistant turns within the current window
        3. Current user turn (waiting for model response)
        
        This ensures:
        - No automatic system prompt insertion
        - Image count matches <image> placeholder count
        - Format matches training data exactly
        
        Window management:
        - Every num_frames steps, reset window state and re-sample history
        - Within window, accumulate dialogue turns
        
        Args:
            env: Habitat environment
            episode: Episode to evaluate
            env_idx: Environment index for multi-env support
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.model.reset_for_env(env_idx)
        
        env.current_episode = episode
        observations = env.reset()
        
        instruction = episode.instruction.instruction_text
        episode_id = episode.episode_id
        scene_id = episode.scene_id.split('/')[-2] if '/' in episode.scene_id else episode.scene_id
        
        # State tracking
        rgb_list = []  # All RGB observations as PIL Images
        action_seq = []  # Pending actions to execute
        step_id = 0
        
        # Video generation
        vis_frames = []
        
        # Window state for complete multi-turn dialogue
        window_start = 0
        window_responses = []  # Cache assistant responses within window
        window_conjunctions = []  # Cache conjunctions used within window
        
        max_steps = self.config.habitat.environment.max_episode_steps
        
        while not env.episode_over and step_id < max_steps:
            # 1. Collect observation
            rgb = observations["rgb"]
            current_img = Image.fromarray(rgb).convert('RGB')
            rgb_list.append(current_img)
            
            # 2. If action queue is empty, predict new actions
            if len(action_seq) == 0:
                # Check if we need to start a new window
                if step_id % self.num_frames == 0:
                    # Reset window state for new window
                    window_start = step_id
                    window_responses = []
                    window_conjunctions = []
                
                try:
                    # Build COMPLETE multi-turn messages (matching training format)
                    messages, images = self.build_complete_messages(
                        instruction=instruction,
                        rgb_list=rgb_list,
                        window_start=window_start,
                        current_step=step_id,
                        window_responses=window_responses,
                        window_conjunctions=window_conjunctions,
                    )
                    
                    # Record the conjunction used for this turn
                    # (extract from the last user message)
                    last_user_content = messages[-1]['content']
                    for conj in self.conjunctions:
                        if last_user_content.startswith(conj):
                            window_conjunctions.append(conj)
                            break
                    else:
                        window_conjunctions.append(self.conjunctions[0])
                    
                    # Encode complete messages with all images
                    encoded = self.template.encode({'messages': messages, 'images': images})
                    input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                    
                    # Prepare model inputs with visual features
                    model_inputs = {'input_ids': input_ids}
                    
                    if 'pixel_values' in encoded:
                        pixel_values = encoded['pixel_values']
                        if isinstance(pixel_values, torch.Tensor):
                            model_inputs['pixel_values'] = pixel_values.to(self.device).to(self.model.dtype)
                        else:
                            model_inputs['pixel_values'] = torch.tensor(pixel_values).to(self.device).to(self.model.dtype)
                    
                    if 'image_grid_thw' in encoded:
                        grid_thw = encoded['image_grid_thw']
                        if isinstance(grid_thw, torch.Tensor):
                            model_inputs['image_grid_thw'] = grid_thw.to(self.device)
                        else:
                            model_inputs['image_grid_thw'] = torch.tensor(grid_thw).to(self.device)
                    
                    # Generate
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        use_cache=True,
                    )
                    
                    # Decode only the generated part
                    generated_ids = outputs[0][input_ids.shape[1]:]
                    output_text = self.processor.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    ).strip()
                    
                    # Cache response for next turn in this window
                    window_responses.append(output_text)
                    
                    # Parse actions
                    action_seq = self.parse_actions(output_text)
                    
                    if self.args.verbose if hasattr(self.args, 'verbose') else False:
                        print(f"Step {step_id}: {output_text} -> {action_seq}")
                    
                except Exception as e:
                    print(f"[Warning] Generation failed at step {step_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    action_seq = []
                
                # Default to STOP if no valid actions
                if not action_seq:
                    action_seq = [0]
            
            # 3. Collect visualization frame (before executing action)
            if self.save_video:
                info = env.get_metrics()
                if info.get('top_down_map') is not None:
                    frame = observations_to_image({'rgb': observations['rgb']}, info)
                    # Add instruction text at the top of the frame
                    frame = self.append_text_to_image(frame, f"Instruction: {instruction}", position='top')
                    vis_frames.append(frame)
            
            # 4. Execute action
            action = action_seq.pop(0)
            observations = env.step(action)
            step_id += 1
        
        # Get final metrics
        metrics = env.get_metrics()
        
        # Save video if requested
        if self.save_video and len(vis_frames) > 0:
            try:
                # Create videos directory only when we actually have frames to save
                video_dir = os.path.join(self.output_path, 'videos')
                os.makedirs(video_dir, exist_ok=True)
                
                # Get success rate from metrics
                success_rate = metrics.get("success", 0.0)
                # Format success rate as percentage (e.g., 1.0 -> "100", 0.5 -> "50")
                success_rate_str = f"{int(success_rate * 100)}"
                
                # Generate video filename: episode_id_success_rate
                video_filename = f"{episode_id}_{success_rate_str}"
                
                # Suppress progress bar and warnings from images_to_video
                # Redirect stdout and stderr to suppress all output
                # Suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Suppress imageio logger warnings
                    imageio_logger = logging.getLogger('imageio')
                    old_level = imageio_logger.level
                    imageio_logger.setLevel(logging.ERROR)
                    
                    try:
                        # Redirect stdout and stderr to suppress tqdm progress bars and other output
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            images_to_video(
                                vis_frames, 
                                video_dir, 
                                video_filename,
                                fps=6,
                                quality=9
                            )
                    finally:
                        # Restore imageio logger level
                        imageio_logger.setLevel(old_level)
            except Exception as e:
                print(f"[Warning] Failed to save video for episode {episode_id}: {e}")
        
        return metrics


def run_evaluation(
    model_path: str,
    habitat_config_path: str,
    eval_split: str = 'val_unseen',
    num_frames: int = 32,
    num_history: int = 8,
    num_future_steps: int = 4,
    output_dir: str = './results/eval',
    save_video: bool = False,
):
    """
    Run VLN evaluation with the given parameters.
    
    This is a convenience function that can be called programmatically.
    """
    import argparse
    
    args = argparse.Namespace(
        model_path=model_path,
        habitat_config_path=habitat_config_path,
        eval_split=eval_split,
        num_frames=num_frames,
        num_history=num_history,
        num_future_steps=num_future_steps,
        output_dir=output_dir,
        save_video=save_video,
    )
    
    # Import here to avoid circular imports
    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.template import TemplateType
    
    # Load model
    model, processor = get_model_tokenizer(
        model_id_or_path=model_path,
        model_type='streamvln_qwen2_5_vl',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    # Get template
    template = get_template(
        template_type=TemplateType.qwen2_5_vl,
        processor=processor,
        model=model
    )
    
    # Initialize model cache
    model.reset(env_num=1)
    
    # Create evaluator
    evaluator = VLNEvaluator(
        config_path=habitat_config_path,
        model=model,
        processor=processor,
        template=template,
        args=args
    )
    
    return evaluator
