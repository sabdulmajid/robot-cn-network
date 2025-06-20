#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from robot_cn_network.envs import create_eval_env
from robot_cn_network.models import ACTPolicy, ModelConfig
from robot_cn_network.utils import (
    setup_logging, load_model, get_device, create_video_from_frames,
    compute_metrics, EpisodeBuffer
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Robot Policy")
    
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--env-id", type=str, default="gym_hil/PandaPickCubeBase-v0")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="./videos/evaluation")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--data-dir", type=str, default="./data/evaluation")
    
    # Analysis arguments
    parser.add_argument(
        "--compare-human",
        type=str,
        default=None,
        help="Path to human demonstration data for comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/evaluation",
        help="Directory to save evaluation results"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    device = get_device(args.device)
    
    logger.info("Starting policy evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Environment: {args.env_id}")
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.record_video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    
    if args.save_data:
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    
    # Create model (we'll need to infer architecture from checkpoint)
    model_config = ModelConfig()  # Default config
    
    # Create a temporary environment to get observation specs
    temp_env = create_eval_env(
        env_id=args.env_id,
        render_mode="rgb_array",
        seed=args.seed
    )
    
    temp_obs, _ = temp_env.reset()
    state_dim = len(temp_obs.get('agent_pos', []))
    camera_names = list(temp_obs.get('pixels', {}).keys())
    temp_env.close()
    
    # Create model
    model = ACTPolicy(
        config=model_config,
        state_dim=state_dim,
        camera_names=camera_names
    )
    
    # Load model weights
    checkpoint_info = load_model(model, args.model_path, device)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from epoch {checkpoint_info.get('epoch', 'unknown')}")
    
    # Create evaluation environment
    env = create_eval_env(
        env_id=args.env_id,
        render_mode=args.render_mode,
        seed=args.seed
    )
    
    logger.info("Starting evaluation episodes...")
    
    # Evaluation statistics
    episode_stats = []
    all_episode_data = []
    
    for episode_idx in range(args.num_episodes):
        logger.info(f"Episode {episode_idx + 1}/{args.num_episodes}")
        
        # Reset environment
        obs, info = env.reset()
        episode_buffer = EpisodeBuffer()
        frames = []
        
        episode_reward = 0
        step_count = 0
        episode_success = False
        
        while step_count < args.max_episode_length:
            # Prepare observation for model
            model_input = prepare_observation(obs, device)
            
            # Get action from model
            with torch.no_grad():
                action = model.predict_action(
                    model_input['images'],
                    model_input['state']
                )
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Store data
            episode_buffer.add_step(obs, action, reward, done, info)
            
            # Record frame if needed
            if args.record_video:
                frame = render_frame(obs)
                if frame is not None:
                    frames.append(frame)
            
            episode_reward += reward
            step_count += 1
            
            # Check for success
            if 'is_success' in info:
                episode_success = info['is_success']
                if episode_success:
                    logger.info(f"Episode {episode_idx + 1} - SUCCESS!")
                    break
            
            if done or truncated:
                episode_success = info.get('is_success', False)
                break
        
        # Episode completed
        episode_data = episode_buffer.get_episode()
        
        # Store statistics
        episode_stat = {
            'episode': episode_idx + 1,
            'success': episode_success,
            'reward': episode_reward,
            'length': step_count,
            'actions': episode_data['actions'],
        }
        episode_stats.append(episode_stat)
        
        if args.save_data:
            all_episode_data.append(episode_data)
        
        # Save video
        if args.record_video and frames:
            video_path = Path(args.video_dir) / f"eval_episode_{episode_idx:03d}.mp4"
            create_video_from_frames(frames, video_path, fps=20)
            logger.info(f"Video saved: {video_path}")
        
        logger.info(
            f"Episode {episode_idx + 1} completed - "
            f"Success: {episode_success}, "
            f"Reward: {episode_reward:.3f}, "
            f"Steps: {step_count}"
        )
    
    env.close()
    
    # Compute overall statistics
    success_rate = np.mean([stat['success'] for stat in episode_stats])
    avg_reward = np.mean([stat['reward'] for stat in episode_stats])
    avg_length = np.mean([stat['length'] for stat in episode_stats])
    
    # Action analysis
    all_actions = np.concatenate([stat['actions'] for stat in episode_stats], axis=0)
    action_stats = analyze_actions(all_actions)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes evaluated: {args.num_episodes}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average episode length: {avg_length:.1f}")
    
    print(f"\nAction Statistics:")
    for key, value in action_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'model_path': args.model_path,
        'env_id': args.env_id,
        'num_episodes': args.num_episodes,
        'success_rate': float(success_rate),
        'average_reward': float(avg_reward),
        'average_length': float(avg_length),
        'episode_stats': episode_stats,
        'action_stats': action_stats,
    }
    
    # Save detailed results
    import json
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Save episode data if requested
    if args.save_data:
        import pickle
        data_file = Path(args.data_dir) / "evaluation_episodes.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(all_episode_data, f)
        logger.info(f"Episode data saved to: {data_file}")
    
    # Compare with human data if provided
    if args.compare_human:
        logger.info("Comparing with human demonstrations...")
        comparison_results = compare_with_human(all_actions, args.compare_human)
        
        print(f"\nComparison with Human Demonstrations:")
        for key, value in comparison_results.items():
            print(f"  {key}: {value:.4f}")
        
        results['human_comparison'] = comparison_results
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    logger.info("Evaluation completed!")


def prepare_observation(obs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Prepare observation for model input."""
    model_input = {
        'images': {},
        'state': None
    }
    
    # Process images
    if 'pixels' in obs:
        for camera_name, image in obs['pixels'].items():
            # Convert to tensor and ensure correct format
            if isinstance(image, np.ndarray):
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                
                if image.ndim == 3 and image.shape[-1] == 3:  # [H, W, C] -> [C, H, W]
                    image = image.transpose(2, 0, 1)
                
                image = torch.FloatTensor(image).unsqueeze(0).to(device)  # Add batch dim
            
            model_input['images'][camera_name] = image
    
    # Process state
    if 'agent_pos' in obs:
        state = torch.FloatTensor(obs['agent_pos']).unsqueeze(0).to(device)  # Add batch dim
        model_input['state'] = state
    else:
        # Default state if not available
        model_input['state'] = torch.zeros(1, 7).to(device)
    
    return model_input


def render_frame(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Render frame for video recording."""
    if 'pixels' in obs:
        front_img = obs['pixels'].get('front')
        wrist_img = obs['pixels'].get('wrist')
        
        if front_img is not None and wrist_img is not None:
            # Ensure images are uint8
            if front_img.dtype != np.uint8:
                front_img = (front_img * 255).astype(np.uint8)
            if wrist_img.dtype != np.uint8:
                wrist_img = (wrist_img * 255).astype(np.uint8)
            
            # Combine views horizontally
            combined_frame = np.concatenate([front_img, wrist_img], axis=1)
            return combined_frame
    
    return None


def analyze_actions(actions: np.ndarray) -> Dict[str, float]:
    """Analyze action statistics."""
    stats = {}
    
    # Basic statistics
    stats['mean_action_magnitude'] = np.mean(np.linalg.norm(actions, axis=1))
    stats['std_action_magnitude'] = np.std(np.linalg.norm(actions, axis=1))
    
    # Per-dimension statistics
    for i in range(actions.shape[1]):
        stats[f'mean_action_dim_{i}'] = np.mean(actions[:, i])
        stats[f'std_action_dim_{i}'] = np.std(actions[:, i])
    
    # Action smoothness (difference between consecutive actions)
    action_diffs = np.diff(actions, axis=0)
    stats['action_smoothness'] = np.mean(np.linalg.norm(action_diffs, axis=1))
    
    return stats


def compare_with_human(
    policy_actions: np.ndarray, 
    human_data_path: str
) -> Dict[str, float]:
    """Compare policy actions with human demonstrations."""
    import pickle
    
    # Load human data
    with open(human_data_path, 'rb') as f:
        human_episodes = pickle.load(f)
    
    # Extract human actions
    human_actions = []
    for episode in human_episodes:
        if isinstance(episode, dict) and 'actions' in episode:
            human_actions.append(episode['actions'])
        elif isinstance(episode, list):
            # Episode is a list of steps
            for step in episode:
                if 'action' in step:
                    human_actions.append(step['action'])
    
    human_actions = np.array(human_actions)
    
    # Compute comparison metrics
    comparison = {}
    
    # Action magnitude comparison
    policy_magnitudes = np.linalg.norm(policy_actions, axis=1)
    human_magnitudes = np.linalg.norm(human_actions, axis=1)
    
    comparison['magnitude_difference'] = abs(
        np.mean(policy_magnitudes) - np.mean(human_magnitudes)
    )
    
    # Action smoothness comparison  
    policy_diffs = np.diff(policy_actions, axis=0)
    human_diffs = np.diff(human_actions, axis=0)
    
    policy_smoothness = np.mean(np.linalg.norm(policy_diffs, axis=1))
    human_smoothness = np.mean(np.linalg.norm(human_diffs, axis=1))
    
    comparison['smoothness_difference'] = abs(policy_smoothness - human_smoothness)
    
    # Per-dimension comparison
    for i in range(min(policy_actions.shape[1], human_actions.shape[1])):
        policy_mean = np.mean(policy_actions[:, i])
        human_mean = np.mean(human_actions[:, i])
        comparison[f'dim_{i}_difference'] = abs(policy_mean - human_mean)
    
    return comparison


if __name__ == "__main__":
    main()
