#!/usr/bin/env python3

import argparse
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import gymnasium as gym
import gym_hil

from robot_cn_network.envs import make_env
from robot_cn_network.utils import setup_logging, EpisodeBuffer, create_video_from_frames
from robot_cn_network.data import DataCollector


def main():
    parser = argparse.ArgumentParser(description="Collect Robot Demonstration Data")
    
    parser.add_argument("--env-id", type=str, default="gym_hil/PandaPickCubeKeyboard-v0")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="./data/demonstrations")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--min-episode-length", type=int, default=10)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--interactive-rating", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="./videos/demonstrations")
    parser.add_argument("--use-gamepad", action="store_true")
    parser.add_argument(
        "--controller-config",
        type=str,
        default=None,
        help="Path to controller configuration file"
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
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data collection")
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Target episodes: {args.num_episodes}")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    if args.record_video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env_kwargs = {}
    if args.controller_config:
        env_kwargs['controller_config_path'] = args.controller_config
    
    if args.use_gamepad:
        # Switch to gamepad environment
        args.env_id = args.env_id.replace("Keyboard", "Gamepad")
    
    env = make_env(
        env_id=args.env_id,
        render_mode=args.render_mode,
        image_obs=True,
        seed=args.seed,
        **env_kwargs
    )
    
    # Create data collector
    data_collector = DataCollector(args.data_dir)
    
    logger.info("Environment created successfully")
    print_instructions()
    input("Press ENTER to start data collection...")
    
    # Data collection statistics
    collected_episodes = 0
    successful_episodes = 0
    total_steps = 0
    
    try:
        while collected_episodes < args.num_episodes:
            logger.info(f"\nStarting episode {collected_episodes + 1}/{args.num_episodes}")
            
            # Reset environment
            obs, info = env.reset()
            episode_buffer = EpisodeBuffer()
            frames = []
            
            episode_reward = 0
            step_count = 0
            episode_success = False
            
            print(f"\n{'='*50}")
            print(f"EPISODE {collected_episodes + 1}")
            print(f"{'='*50}")
            print("Starting episode... Take control of the robot!")
            
            # Episode loop
            while step_count < args.episode_length:
                # Environment handles human input
                action = env.action_space.sample()  # Placeholder
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                
                # Store step data
                episode_buffer.add_step(obs, action, reward, done, info)
                
                # Record video frame if needed
                if args.record_video:
                    if 'pixels' in obs:
                        front_img = obs['pixels'].get('front')
                        wrist_img = obs['pixels'].get('wrist')
                        
                        if front_img is not None and wrist_img is not None:
                            # Ensure images are uint8
                            if front_img.dtype != np.uint8:
                                front_img = (front_img * 255).astype(np.uint8)
                            if wrist_img.dtype != np.uint8:
                                wrist_img = (wrist_img * 255).astype(np.uint8)
                            
                            # Combine views
                            combined_frame = np.concatenate([front_img, wrist_img], axis=1)
                            frames.append(combined_frame)
                
                episode_reward += reward
                step_count += 1
                
                # Check for success/failure
                if 'is_success' in info:
                    episode_success = info['is_success']
                    if episode_success:
                        print("SUCCESS! Episode completed successfully!")
                        break
                
                if done or truncated:
                    episode_success = info.get('is_success', False)
                    break
                
                # Small delay for human control
                time.sleep(0.01)
            
            # Episode completed
            episode_data = episode_buffer.get_episode()
            
            print(f"\nEpisode completed!")
            print(f"Steps: {step_count}")
            print(f"Reward: {episode_reward:.3f}")
            print(f"Success: {episode_success}")
            
            # Decide whether to save episode
            save_episode = True
            
            # Check minimum length
            if step_count < args.min_episode_length:
                print(f"Episode too short ({step_count} < {args.min_episode_length}), discarding...")
                save_episode = False
            
            # Check success requirement
            if args.success_only and not episode_success:
                print("Episode not successful, discarding...")
                save_episode = False
            
            # Interactive rating
            if save_episode and args.interactive_rating:
                rating = input("Rate this episode (1-5, or 0 to discard): ")
                try:
                    rating_score = int(rating)
                    if rating_score == 0:
                        save_episode = False
                        print("Episode discarded by user")
                    elif rating_score < 3:
                        user_confirm = input("Low rating. Save anyway? (y/n): ").lower()
                        save_episode = user_confirm == 'y'
                    episode_data['rating'] = rating_score
                except ValueError:
                    print("Invalid rating, saving episode anyway")
            
            # Save episode if approved
            if save_episode:
                # Add metadata
                episode_data['metadata'] = {
                    'episode_id': collected_episodes,
                    'success': episode_success,
                    'total_reward': episode_reward,
                    'length': step_count,
                    'timestamp': time.time(),
                }
                
                # Save episode data
                episode_file = data_collector.save_episode(episode_data['observations'])
                logger.info(f"Saved episode data: {episode_file}")
                
                # Save video if recorded
                if args.record_video and frames:
                    video_path = Path(args.video_dir) / f"episode_{collected_episodes:04d}.mp4"
                    create_video_from_frames(frames, video_path, fps=20)
                    logger.info(f"Saved video: {video_path}")
                
                collected_episodes += 1
                total_steps += step_count
                
                if episode_success:
                    successful_episodes += 1
                
                # Print statistics
                success_rate = successful_episodes / collected_episodes * 100
                avg_steps = total_steps / collected_episodes
                
                print(f"\n--- Collection Statistics ---")
                print(f"Episodes collected: {collected_episodes}/{args.num_episodes}")
                print(f"Success rate: {success_rate:.1f}%")
                print(f"Average episode length: {avg_steps:.1f}")
                print(f"Total steps collected: {total_steps}")
            
            # Ask if user wants to continue
            if collected_episodes < args.num_episodes:
                continue_choice = input("\nContinue to next episode? (y/n/q): ").lower()
                if continue_choice == 'q':
                    break
                elif continue_choice == 'n':
                    # Allow user to redo current episode
                    redo = input("Redo this episode? (y/n): ").lower()
                    if redo != 'y':
                        break
    
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    
    finally:
        env.close()
        logger.info("Environment closed")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("DATA COLLECTION COMPLETED")
    print(f"{'='*60}")
    print(f"Episodes collected: {collected_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {successful_episodes/max(collected_episodes, 1)*100:.1f}%")
    print(f"Total steps: {total_steps}")
    print(f"Average episode length: {total_steps/max(collected_episodes, 1):.1f}")
    print(f"Data saved in: {args.data_dir}")
    
    if args.record_video:
        print(f"Videos saved in: {args.video_dir}")
    
    # Get dataset statistics
    dataset_stats = data_collector.get_dataset_stats()
    print(f"\nDataset Statistics:")
    for key, value in dataset_stats.items():
        print(f"  {key}: {value}")
    
    logger.info("Data collection session completed")


def print_instructions():
    """Print data collection instructions."""
    print("\n" + "="*60)
    print("ROBOT DEMONSTRATION DATA COLLECTION")
    print("="*60)
    print("Instructions:")
    print("1. You will control the robot to demonstrate the task")
    print("2. Try to complete the task successfully")
    print("3. Make smooth, natural movements")
    print("4. Each episode will be saved automatically")
    print("5. You can rate episodes if interactive rating is enabled")
    print("\nControls:")
    print("  Arrow Keys    - Move in X-Y plane")
    print("  Shift + ↑↓    - Move up/down in Z axis")
    print("  Space         - Enable/disable control")
    print("  Ctrl          - Close/open gripper")
    print("  R             - Reset episode")
    print("  ESC           - Exit")
    print("\nTips:")
    print("- Focus on task completion over speed")
    print("- Demonstrate diverse approaches")
    print("- Reset if episode goes poorly")
    print("="*60)


if __name__ == "__main__":
    main()
