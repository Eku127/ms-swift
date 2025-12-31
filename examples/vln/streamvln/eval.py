# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Habitat Evaluation Entry Point

This script runs VLN evaluation using the trained StreamVLNQwen25VL model
in Habitat environment. Supports both single-process and distributed modes.

Usage:
    # Single-process evaluation
    python -m examples.vln.streamvln.eval --model_path /path/to/checkpoint
    
    # Distributed evaluation (8 GPUs)
    torchrun --nproc_per_node=8 -m examples.vln.streamvln.eval \
        --model_path /path/to/checkpoint --distributed
"""

import os
import sys
import json
import argparse
import zipfile
import torch
import torch.distributed as dist
import tqdm

# Setup paths for both module and direct execution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_msswift_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
if _msswift_root not in sys.path:
    sys.path.insert(0, _msswift_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def init_distributed():
    """Initialize distributed environment if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if world_size > 1:
            # Set longer NCCL timeout to avoid timeout issues
            import datetime
            timeout = datetime.timedelta(hours=2)
            dist.init_process_group(backend='nccl', timeout=timeout)
            torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    
    return 0, 1, 0


def main():
    parser = argparse.ArgumentParser(description="StreamVLN Habitat Evaluation")
    
    # Model and data
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained checkpoint")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml', 
                        help="Path to habitat yaml (relative to this script)")
    parser.add_argument("--eval_split", type=str, default='val_unseen', 
                        help="Dataset split to evaluate")
    
    # VLN parameters
    parser.add_argument("--num_frames", type=int, default=32, 
                        help="Streaming window size")
    parser.add_argument("--num_history", type=int, default=8, 
                        help="Number of history frames to sample")
    parser.add_argument("--num_future_steps", type=int, default=4, 
                        help="Number of actions to predict per step")
    
    # Output
    parser.add_argument("--output_dir", type=str, default='./results/eval', 
                        help="Output directory for results")
    parser.add_argument("--save_video", action="store_true", 
                        help="Save visualization videos")
    parser.add_argument("--video_compression", action="store_true",
                        help="Compress videos into zip files (400 videos per zip) and delete source files")
    
    # Execution mode
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed evaluation (use with torchrun)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to evaluate (for debugging)")
    
    args = parser.parse_args()
    
    # Initialize distributed if requested and available
    if args.distributed:
        rank, world_size, local_rank = init_distributed()
    else:
        rank, world_size, local_rank = 0, 1, 0
    
    is_main = (rank == 0)
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        if world_size > 1:
            print(f"[Distributed Mode] {world_size} processes")
        else:
            print("[Single Process Mode]")
    
    # Ensure directories are created before other ranks proceed
    if world_size > 1:
        dist.barrier()
    
    # Import StreamVLN module to register the model
    try:
        import examples.vln.streamvln
    except ImportError:
        pass
    
    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.template import TemplateType
    
    try:
        from examples.vln.streamvln.evaluator import VLNEvaluator
    except ImportError:
        from evaluator import VLNEvaluator
    
    # Load Model
    if is_main:
        print(f"Loading model from {args.model_path}...")
    
    # Device mapping based on mode
    if world_size > 1:
        device_map = {'': local_rank}  # Each process uses its own GPU
    else:
        device_map = 'auto'  # Single process can use model parallelism
    
    model, processor = get_model_tokenizer(
        model_id_or_path=args.model_path,
        model_type='streamvln_qwen2_5_vl',
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_impl='flash_attn',  # Use flash attention for faster inference
    )
    
    template = get_template(
        template_type=TemplateType.qwen2_5_vl,
        processor=processor
    )
    
    # Initialize model cache
    model.reset(env_num=1)
    
    # Habitat config path
    habitat_config_path = args.habitat_config_path
    if not os.path.isabs(habitat_config_path):
        # Try relative to current script dir first
        path_wrt_script = os.path.join(_current_dir, habitat_config_path)
        # Try relative to project root next
        path_wrt_root = os.path.join(_msswift_root, habitat_config_path)
        
        if os.path.exists(path_wrt_script):
            habitat_config_path = path_wrt_script
        elif os.path.exists(path_wrt_root):
            habitat_config_path = path_wrt_root
        else:
            # Default to script dir if neither exists (will throw error later)
            habitat_config_path = path_wrt_script
    
    # Create evaluator
    evaluator = VLNEvaluator(
        config_path=habitat_config_path,
        model=model,
        processor=processor,
        template=template,
        args=args
    )
    
    # Create environment and get episodes
    env = evaluator.config_env()
    all_episodes = env.episodes
    
    if args.max_episodes is not None:
        all_episodes = all_episodes[:args.max_episodes]
    
    # Group episodes by scene to minimize scene loading overhead (like StreamVLN original)
    # Then use interleaved sharding within each scene
    scene_episode_dict = {}
    for episode in all_episodes:
        scene_id = episode.scene_id
        if scene_id not in scene_episode_dict:
            scene_episode_dict[scene_id] = []
        scene_episode_dict[scene_id].append(episode)
    
    # Build episode list for this rank: process all assigned episodes from one scene before moving to next
    my_episodes = []
    for scene_id in sorted(scene_episode_dict.keys()):
        scene_episodes = scene_episode_dict[scene_id]
        if world_size > 1:
            # Interleaved sharding within each scene
            my_episodes.extend(scene_episodes[rank::world_size])
        else:
            my_episodes.extend(scene_episodes)
    
    if is_main:
        print(f"Environment: {args.eval_split}, Total: {len(all_episodes)}, "
              f"This process: {len(my_episodes)}, Scenes: {len(scene_episode_dict)}")
    
    # Evaluation loop
    results = []
    success_count = 0
    
    # Progress bar: show on rank 0 with total info, other ranks print periodically
    if world_size > 1:
        desc = f"Rank 0 ({len(my_episodes)} eps, {world_size} GPUs total)"
    else:
        desc = "Evaluating"
    pbar = tqdm.tqdm(my_episodes, desc=desc, disable=not is_main)
    
    for i, episode in enumerate(pbar):
        # Extract scene name (without path, without .glb extension)
        scene_name = os.path.basename(episode.scene_id).replace('.glb', '')
        # Get instruction text
        instruction = getattr(episode, 'instruction', {})
        if isinstance(instruction, dict):
            instruction_text = instruction.get('instruction_text', '')
        else:
            instruction_text = str(instruction) if instruction else ''
        
        try:
            metrics = evaluator.eval_episode(env, episode, env_idx=0)
            
            result = {
                "episode_id": episode.episode_id,
                "scene_id": scene_name,
                "success": float(metrics.get("success", 0)),
                "spl": float(metrics.get("spl", 0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0)),
                "oracle_success": float(metrics.get("oracle_success", 0)),
                "instruction": instruction_text,
            }
            if world_size > 1:
                result["rank"] = rank
                
        except Exception as e:
            print(f"\n[Rank {rank}] Error on episode {episode.episode_id}: {e}")
            result = {
                "episode_id": episode.episode_id,
                "scene_id": scene_name,
                "success": 0.0,
                "spl": 0.0,
                "distance_to_goal": float('inf'),
                "oracle_success": 0.0,
                "error": str(e),
                "instruction": instruction_text,
            }
        
        results.append(result)
        
        if result["success"]:
            success_count += 1
        
        # Save intermediate results
        if is_main and (i + 1) % 10 == 0:
            with open(os.path.join(args.output_dir, "results_partial.json"), "w") as f:
                json.dump(results, f, indent=2)
    
    env.close()
    
    # Aggregate results using dist.all_gather (like StreamVLN original)
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
        
        # Convert results to tensors for all_gather
        # Extract metrics as tensors
        sucs = torch.tensor([r.get("success", 0.0) for r in results], dtype=torch.float32, device=device)
        spls = torch.tensor([r.get("spl", 0.0) for r in results], dtype=torch.float32, device=device)
        oss = torch.tensor([r.get("oracle_success", 0.0) for r in results], dtype=torch.float32, device=device)
        nes = torch.tensor([r.get("distance_to_goal", 0.0) for r in results], dtype=torch.float32, device=device)
        ep_num = torch.tensor(len(results), dtype=torch.int64, device=device)
        
        # First, gather episode counts from all ranks
        ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
        dist.all_gather(ep_num_all, ep_num)
        
        # Prepare tensors for gathering (with correct sizes)
        sucs_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        spls_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        oss_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        nes_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        
        # Synchronize before gathering results
        dist.barrier()
        
        # Gather all results
        dist.all_gather(sucs_all, sucs)
        dist.all_gather(spls_all, spls)
        dist.all_gather(oss_all, oss)
        dist.all_gather(nes_all, nes)
        
        dist.barrier()
        
        # Concatenate results on main process
        if is_main:
            sucs_all = torch.cat(sucs_all, dim=0)
            spls_all = torch.cat(spls_all, dim=0)
            oss_all = torch.cat(oss_all, dim=0)
            nes_all = torch.cat(nes_all, dim=0)
            
            # Also save individual rank results for debugging
            rank_file = os.path.join(args.output_dir, f"results_rank{rank}.json")
            with open(rank_file, "w") as f:
                json.dump(results, f, indent=2)
    else:
        # Single process mode
        sucs_all = torch.tensor([r.get("success", 0.0) for r in results])
        spls_all = torch.tensor([r.get("spl", 0.0) for r in results])
        oss_all = torch.tensor([r.get("oracle_success", 0.0) for r in results])
        nes_all = torch.tensor([r.get("distance_to_goal", 0.0) for r in results])
    
    # Summary (main process only)
    if is_main:
        # Use gathered tensor results (like StreamVLN original)
        total_episodes = len(sucs_all)
        success_rate = (sucs_all.sum() / total_episodes).item() if total_episodes > 0 else 0
        mean_spl = (spls_all.sum() / total_episodes).item() if total_episodes > 0 else 0
        mean_os = (oss_all.sum() / total_episodes).item() if total_episodes > 0 else 0
        # Filter out invalid navigation errors (inf values stored as 0 or very large)
        valid_nes = nes_all[nes_all < 1000]  # Assume distance > 1000m is invalid
        mean_ne = valid_nes.mean().item() if len(valid_nes) > 0 else 0
        
        summary = {
            "eval_split": args.eval_split,
            "success_rate": success_rate,
            "mean_spl": mean_spl,
            "oracle_success": mean_os,
            "navigation_error": mean_ne,
            "total_episodes": total_episodes,
            "world_size": world_size,
            "model_path": args.model_path,
            "num_frames": args.num_frames,
            "num_history": args.num_history,
        }
        
        print(f"\n" + "="*50)
        print(f"Evaluation Summary ({args.eval_split})")
        print(f"="*50)
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Mean SPL: {summary['mean_spl']:.4f}")
        print(f"Oracle Success: {summary['oracle_success']:.2%}")
        print(f"Navigation Error: {summary['navigation_error']:.2f}m")
        print(f"Total Episodes: {total_episodes}")
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs")
        print(f"="*50)
        
        with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results (only from rank 0's local results in distributed mode)
        # Full aggregated results are in the tensor summaries above
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            f.write("[\n")
            for idx, r in enumerate(results):
                line = json.dumps(r, ensure_ascii=False)
                if idx < len(results) - 1:
                    f.write(f"  {line},\n")
                else:
                    f.write(f"  {line}\n")
            f.write("]\n")
        
        print(f"Results saved to {args.output_dir}")
        
        # Video compression logic
        if args.video_compression and args.save_video:
            video_dir = os.path.join(args.output_dir, "videos")
            if os.path.exists(video_dir):
                print("Compressing videos...")
                chunk_size = 400
                for i in range(0, len(all_episodes), chunk_size):
                    start_idx = i
                    end_idx = min(i + chunk_size, len(all_episodes))
                    
                    # Naming logic: 0-400, 401-800, etc.
                    if start_idx == 0:
                        zip_name = f"0-{end_idx}.zip"
                    else:
                        zip_name = f"{start_idx+1}-{end_idx}.zip"
                    
                    zip_path = os.path.join(video_dir, zip_name)
                    
                    added_count = 0
                    # Get all mp4 files in the directory once to speed up lookup
                    all_video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for j in range(start_idx, end_idx):
                            ep_id = str(all_episodes[j].episode_id)
                            # Match files like "{ep_id}.mp4" or "{ep_id}_{success}.mp4"
                            matched_files = [f for f in all_video_files 
                                           if f == f"{ep_id}.mp4" or f.startswith(f"{ep_id}_")]
                            
                            for video_file in matched_files:
                                video_path = os.path.join(video_dir, video_file)
                                if os.path.exists(video_path):
                                    zf.write(video_path, video_file)
                                    os.remove(video_path)
                                    added_count += 1
                    
                    if added_count > 0:
                        print(f"Created {zip_name} with {added_count} videos.")
                    else:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                print("Video compression completed.")
    
    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
