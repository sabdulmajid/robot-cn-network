#!/usr/bin/env python3

import argparse
import logging
import time
import numpy as np
from pathlib import Path

import gymnasium as gym
import gym_hil

from robot_cn_network.envs import make_env, DataCollectionWrapper
from robot_cn_network.utils import setup_logging, EpisodeBuffer, create_video_from_frames


def main():
    parser = argparse.ArgumentParser(description="Robot Teleoperation Interface")
    
    parser.add_argument("--env-id", type=str, default="gym_hil/PandaPickCubeKeyboard-v0")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-gamepad", action="store_true")
    parser.add_argument("--controller-config", type=str, default=None)
    parser.add_argument("--collect-data", action="store_true")
    parser.add_argument("--data-dir", type=str, default="./data/demonstrations")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="./videos")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting robot teleoperation interface")
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Data collection: {args.collect_data}")
    
    env_kwargs = {}
    if args.controller_config:
        env_kwargs['controller_config_path'] = args.controller_config
    
    if args.use_gamepad:
        args.env_id = args.env_id.replace("Keyboard", "Gamepad")
    
    env = make_env(
        env_id=args.env_id,
        render_mode=args.render_mode,
        image_obs=True,
        seed=args.seed,
        **env_kwargs
    )
    
    if args.collect_data:
        env = DataCollectionWrapper(env, save_path=args.data_dir, episode_length=args.episode_length)
    
    if args.record_video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment created successfully")
    print_controls()
    
    try:
        episode_count = 0
        
        while episode_count < args.num_episodes:
            logger.info(f"Starting episode {episode_count + 1}/{args.num_episodes}")
            
            obs, info = env.reset()
            episode_buffer = EpisodeBuffer()
            frames = []
            
            episode_reward = 0
            step_count = 0
            
            while step_count < args.episode_length:
                action = env.action_space.sample()  # gym_hil handles input
                obs, reward, done, truncated, info = env.step(action)
                
                if args.record_video and args.render_mode == "rgb_array":
                    if 'pixels' in obs:
                        front_img = obs['pixels'].get('front', np.zeros((480, 640, 3), dtype=np.uint8))
                        wrist_img = obs['pixels'].get('wrist', np.zeros((480, 640, 3), dtype=np.uint8))
                        combined_frame = np.concatenate([front_img, wrist_img], axis=1)
                        frames.append(combined_frame)
                
                episode_reward += reward
                step_count += 1
                
                if done or truncated:
                    break
                
                if args.render_mode == "human":
                    time.sleep(0.01)
            
            logger.info(f"Episode {episode_count + 1} completed")
            logger.info(f"Steps: {step_count}, Reward: {episode_reward:.3f}")
            
            if args.record_video and frames:
                video_path = Path(args.video_dir) / f"episode_{episode_count:04d}.mp4"
                create_video_from_frames(frames, video_path, fps=20)
                logger.info(f"Video saved: {video_path}")
            
            episode_count += 1
            
            if episode_count < args.num_episodes:
                user_input = input("Continue to next episode? (y/n): ").lower()
                if user_input != 'y':
                    break
    
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted by user")
    
    finally:
        env.close()
        logger.info("Environment closed")
    
    logger.info("Teleoperation session completed")


def print_controls():
    print("\n" + "="*50)
    print("ROBOT TELEOPERATION CONTROLS")
    print("="*50)
    print("Keyboard: Arrow keys (move), Space (enable), Ctrl (gripper), R (reset), ESC (exit)")
    print("Gamepad: Left stick (move), Right stick ↑↓ (Z), RB (enable), LT/RT (gripper)")
    print("="*50)
    print("Press ENTER to start...")
    input()


if __name__ == "__main__":
    main()
