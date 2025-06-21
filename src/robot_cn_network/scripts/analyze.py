#!/usr/bin/env python3

import argparse
import logging
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from robot_cn_network.utils import (
    setup_logging, load_config, TrainingVisualizer, RobotTrajectoryVisualizer, PLOTLY_AVAILABLE
)

def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze robot training results and trajectories")
    
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data directory or trajectory file")
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis",
                        help="Directory to save visualizations")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name of experiment for visualization")
    parser.add_argument("--mode", type=str, default="training",
                        choices=["training", "trajectory", "policy-comparison", "action-space"],
                        help="Visualization mode")
    parser.add_argument("--policy-paths", type=str, nargs="+", default=None,
                        help="Paths to policies for comparison (required for policy-comparison mode)")
    parser.add_argument("--env-id", type=str, default="gym_hil/PandaPickCubeKeyboard-v0",
                        help="Environment ID for policy comparison")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes for policy comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--metric", type=str, default="all",
                        help="Specific metric to visualize (for training mode)")
    parser.add_argument("--interactive", action="store_true",
                        help="Generate interactive plots (requires plotly)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    if not PLOTLY_AVAILABLE and args.interactive:
        logger.warning("Plotly not available, falling back to static plots")
        args.interactive = False
    
    if args.mode == "training":
        visualize_training(args, logger)
    elif args.mode == "trajectory":
        visualize_trajectory(args, logger)
    elif args.mode == "policy-comparison":
        compare_policies(args, logger)
    elif args.mode == "action-space":
        visualize_action_space(args, logger)
    else:
        logger.error(f"Unknown visualization mode: {args.mode}")

def visualize_training(args, logger):
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if data_path.is_file() and data_path.suffix == '.json':
        # Single metrics file
        with open(data_path, 'r') as f:
            metrics_data = json.load(f)
        
        experiment_name = args.experiment_name or data_path.stem
        visualizer = TrainingVisualizer(args.output_dir, experiment_name)
        
        if "config" in metrics_data:
            visualizer.log_config(metrics_data["config"])
        
        metrics = []
        if args.metric != "all":
            metrics = [args.metric]
        
        fig = visualizer.plot_learning_curves(metrics=metrics, interactive=args.interactive)
        
        if args.interactive and fig:
            print(f"Interactive learning curves saved to: {output_dir / f'{experiment_name}_learning_curves.html'}")
        elif fig:
            print(f"Learning curves saved to: {output_dir / f'{experiment_name}_learning_curves.png'}")
        
    elif data_path.is_dir():
        # Directory with multiple runs
        metrics_files = list(data_path.glob("*_metrics.json"))
        if not metrics_files:
            logger.error(f"No metrics files found in: {data_path}")
            return
        
        logger.info(f"Found {len(metrics_files)} metrics files")
        
        # Process each file
        for metrics_file in metrics_files:
            experiment_name = metrics_file.stem.replace("_metrics", "")
            logger.info(f"Processing experiment: {experiment_name}")
            
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            visualizer = TrainingVisualizer(args.output_dir, experiment_name)
            if "config" in metrics_data:
                visualizer.log_config(metrics_data["config"])
                
            metrics = []
            if args.metric != "all":
                metrics = [args.metric]
            
            fig = visualizer.plot_learning_curves(metrics=metrics, interactive=args.interactive)
            
            if args.interactive and fig:
                print(f"Interactive learning curves saved to: {output_dir / f'{experiment_name}_learning_curves.html'}")
            elif fig:
                print(f"Learning curves saved to: {output_dir / f'{experiment_name}_learning_curves.png'}")
    else:
        logger.error(f"Unsupported data path: {data_path}")

def visualize_trajectory(args, logger):
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = RobotTrajectoryVisualizer(args.output_dir)
    
    if data_path.is_file() and data_path.suffix in ['.npy', '.npz']:
        if data_path.suffix == '.npy':
            positions = np.load(data_path)
            if positions.shape[1] != 3:
                logger.error(f"Expected trajectory positions with shape (n, 3), got {positions.shape}")
                return
                
            title = args.experiment_name or f"Trajectory_{data_path.stem}"
            fig = visualizer.visualize_end_effector_path(positions, title=title)
            
            print(f"Trajectory visualization saved to: {output_dir / f'{title.lower().replace(' ', '_')}.html'}")
            
        elif data_path.suffix == '.npz':
            data = np.load(data_path)
            if 'positions' not in data:
                logger.error("Expected 'positions' array in NPZ file")
                return
                
            positions = data['positions']
            orientations = data.get('orientations', None)
            cube_positions = data.get('cube_positions', None)
            
            if args.interactive:
                fig = visualizer.create_trajectory_animation(
                    positions, orientations, cube_positions
                )
                print(f"Trajectory animation saved to: {output_dir / 'trajectory_animation.html'}")
            else:
                title = args.experiment_name or f"Trajectory_{data_path.stem}"
                fig = visualizer.visualize_end_effector_path(
                    positions, 
                    targets=cube_positions if cube_positions is not None else None,
                    title=title
                )
                print(f"Trajectory visualization saved to: {output_dir / f'{title.lower().replace(' ', '_')}.html'}")
    
    elif data_path.is_dir():
        # Directory with multiple trajectory files
        trajectory_files = list(data_path.glob("*.npy")) + list(data_path.glob("*.npz"))
        if not trajectory_files:
            logger.error(f"No trajectory files found in: {data_path}")
            return
            
        logger.info(f"Found {len(trajectory_files)} trajectory files")
        
        for traj_file in trajectory_files:
            if traj_file.suffix == '.npy':
                positions = np.load(traj_file)
                if positions.shape[1] != 3:
                    logger.warning(f"Skipping {traj_file}: Expected positions with shape (n, 3), got {positions.shape}")
                    continue
                    
                title = f"Trajectory_{traj_file.stem}"
                fig = visualizer.visualize_end_effector_path(positions, title=title)
                
            elif traj_file.suffix == '.npz':
                data = np.load(traj_file)
                if 'positions' not in data:
                    logger.warning(f"Skipping {traj_file}: 'positions' array not found")
                    continue
                    
                positions = data['positions']
                orientations = data.get('orientations', None)
                cube_positions = data.get('cube_positions', None)
                
                if args.interactive:
                    fig = visualizer.create_trajectory_animation(
                        positions, orientations, cube_positions
                    )
                else:
                    title = f"Trajectory_{traj_file.stem}"
                    fig = visualizer.visualize_end_effector_path(
                        positions, 
                        targets=cube_positions if cube_positions is not None else None,
                        title=title
                    )
            
            logger.info(f"Processed trajectory file: {traj_file.name}")
    else:
        logger.error(f"Unsupported data path: {data_path}")

def compare_policies(args, logger):
    if not args.policy_paths:
        logger.error("Policy paths are required for policy comparison mode")
        return
        
    for policy_path in args.policy_paths:
        if not Path(policy_path).exists():
            logger.error(f"Policy path does not exist: {policy_path}")
            return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = TrainingVisualizer(args.output_dir, args.experiment_name or "policy_comparison")
    
    try:
        fig, results = visualizer.compare_policies(
            args.policy_paths,
            args.env_id,
            num_episodes=args.num_episodes,
            episode_length=100,
            seed=args.seed
        )
        
        print(f"Policy comparison results saved to: {output_dir / 'policy_comparison.json'}")
        print(f"Policy comparison visualization saved to: {output_dir / 'policy_comparison.html'}")
        
        # Print summary of results
        print("\nPolicy Comparison Summary:")
        print("-" * 50)
        print(f"{'Policy':<20} {'Mean Return':<15} {'Success Rate':<15}")
        print("-" * 50)
        
        for policy_name, result in results.items():
            print(f"{policy_name:<20} {result['mean_return']:<15.4f} {result['success_rate']:<15.4f}")
            
    except ImportError as e:
        logger.error(f"Failed to compare policies: {e}")
        logger.error("Make sure the required packages are installed")

def visualize_action_space(args, logger):
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = TrainingVisualizer(args.output_dir, args.experiment_name or "action_space")
    
    if data_path.is_file() and data_path.suffix in ['.npy', '.npz', '.json']:
        if data_path.suffix == '.npy':
            actions = np.load(data_path)
        elif data_path.suffix == '.npz':
            data = np.load(data_path)
            if 'actions' not in data:
                logger.error("Expected 'actions' array in NPZ file")
                return
            actions = data['actions']
        elif data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
            if 'actions' not in data:
                logger.error("Expected 'actions' key in JSON file")
                return
            actions = np.array(data['actions'])
        
        fig = visualizer.visualize_action_space(actions)
        print(f"Action space visualization saved to: {output_dir / f'{visualizer.experiment_name}_action_space.html'}")
        
    elif data_path.is_dir():
        # Find all action files
        action_files = (
            list(data_path.glob("*actions*.npy")) + 
            list(data_path.glob("*actions*.npz")) +
            list(data_path.glob("*actions*.json"))
        )
        
        if not action_files:
            logger.error(f"No action files found in: {data_path}")
            return
            
        logger.info(f"Found {len(action_files)} action files")
        
        for action_file in action_files:
            if action_file.suffix == '.npy':
                actions = np.load(action_file)
            elif action_file.suffix == '.npz':
                data = np.load(action_file)
                if 'actions' not in data:
                    logger.warning(f"Skipping {action_file}: 'actions' array not found")
                    continue
                actions = data['actions']
            elif action_file.suffix == '.json':
                with open(action_file, 'r') as f:
                    data = json.load(f)
                if 'actions' not in data:
                    logger.warning(f"Skipping {action_file}: 'actions' key not found")
                    continue
                actions = np.array(data['actions'])
                
            experiment_name = f"action_space_{action_file.stem}"
            vis = TrainingVisualizer(args.output_dir, experiment_name)
            fig = vis.visualize_action_space(actions)
            logger.info(f"Processed action file: {action_file.name}")
            print(f"Action space visualization saved to: {output_dir / f'{experiment_name}_action_space.html'}")
    else:
        logger.error(f"Unsupported data path: {data_path}")

if __name__ == "__main__":
    main()
